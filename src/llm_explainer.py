"""
LLM Explainer Module
====================
Takes detected anomalies from the ML pipeline and generates
plain English explanations using a local Llama model via Ollama.
"""

import ollama
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

NORMAL_RANGES = {
    'Pressure': {'min': -0.5, 'max': 0.7, 'unit': 'bar'},
    'Volume Flow RateRMS': {'min': 30.5, 'max': 33.5, 'unit': 'L/min'},
    'Current': {'min': 0.4, 'max': 1.6, 'unit': 'A'},
    'Temperature': {'min': 71.0, 'max': 77.0, 'unit': 'C'}
}

SENSOR_DESCRIPTIONS = {
    'Pressure': 'system pressure',
    'Volume Flow RateRMS': 'water flow rate',
    'Current': 'motor current draw',
    'Temperature': 'fluid temperature'
}


def build_context(anomaly_row, normal_df, anomaly_duration_seconds):
    sensors = ['Pressure', 'Volume Flow RateRMS', 'Current', 'Temperature']
    context_lines = []
    context_lines.append(f"ANOMALY DETECTED at {anomaly_row.name}")
    context_lines.append(f"Duration: {anomaly_duration_seconds} seconds")
    context_lines.append("")
    context_lines.append("SENSOR READINGS AT TIME OF ANOMALY:")
    context_lines.append("-" * 40)

    deviating_sensors = []

    for sensor in sensors:
        if sensor not in anomaly_row.index:
            continue

        current_value = anomaly_row[sensor]
        normal_range = NORMAL_RANGES[sensor]
        description = SENSOR_DESCRIPTIONS[sensor]

        normal_mean = normal_df[sensor].mean() if sensor in normal_df.columns else (
            normal_range['min'] + normal_range['max']) / 2
        deviation = current_value - normal_mean
        deviation_pct = abs(deviation / normal_mean * 100) if normal_mean != 0 else 0

        is_deviating = (current_value < normal_range['min'] or
                        current_value > normal_range['max'])

        if is_deviating:
            status = "ABNORMAL"
            deviating_sensors.append(sensor)
        else:
            status = "Normal"

        context_lines.append(f"{sensor} ({description}):")
        context_lines.append(f"  Current: {current_value:.3f} {normal_range['unit']} [{status}]")
        context_lines.append(f"  Normal range: {normal_range['min']} to {normal_range['max']} {normal_range['unit']}")
        context_lines.append(f"  Deviation: {deviation:+.3f} ({deviation_pct:.1f}% from normal)")
        context_lines.append("")

    context_lines.append(f"PRIMARY AFFECTED SENSORS: {', '.join(deviating_sensors)}")
    return "\n".join(context_lines)


def build_prompt(context):
    system_prompt = """You are an expert industrial engineer specialising in 
pump and valve systems. You analyse sensor anomaly data and provide 
clear, actionable explanations for maintenance engineers.

When given anomaly data, respond in this exact JSON format:
{
    "cause": "One sentence explaining what likely caused this anomaly",
    "severity": "HIGH or MEDIUM or LOW",
    "severity_reason": "One sentence explaining why you chose this severity",
    "action": "Specific action the engineer should take right now",
    "affected_sensors": ["list", "of", "deviating", "sensors"],
    "confidence": "HIGH or MEDIUM or LOW",
    "confidence_reason": "One sentence explaining your confidence level"
}

Rules:
- Base your explanation ONLY on the sensor data provided
- Be specific — mention actual sensor values
- Use simple language a maintenance engineer can act on immediately
- Respond with ONLY the JSON object, no other text"""

    few_shot_example = """Example anomaly:
Pressure: -0.82 bar [ABNORMAL]
Volume Flow RateRMS: 30.1 L/min [ABNORMAL]
Current: 0.87 A [Normal]
Temperature: 74.3 C [Normal]

Example response:
{
    "cause": "Simultaneous pressure collapse and flow reduction suggests valve blockage",
    "severity": "HIGH",
    "severity_reason": "Pressure dropped 40% below normal and persisted for 47 seconds",
    "action": "Inspect valve V1 immediately for debris or mechanical obstruction",
    "affected_sensors": ["Pressure", "Volume Flow RateRMS"],
    "confidence": "HIGH",
    "confidence_reason": "Classic valve fault signature with simultaneous pressure and flow deviation"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_example},
        {"role": "assistant", "content": '{"acknowledged": "understood"}'},
        {"role": "user", "content": f"Now analyse this real anomaly:\n\n{context}"}
    ]

    return messages


def explain_anomaly(anomaly_row, normal_df,
                    anomaly_duration_seconds=30,
                    model="llama3.2"):
    logger.info(f"Generating explanation for anomaly at {anomaly_row.name}")

    context = build_context(anomaly_row, normal_df, anomaly_duration_seconds)
    messages = build_prompt(context)

    logger.info(f"Calling {model}...")

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 500}
        )
        raw_response = response['message']['content']
        logger.info("LLM response received")

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return {
            "cause": "LLM unavailable",
            "severity": "UNKNOWN",
            "severity_reason": "Could not generate explanation",
            "action": "Check Ollama is running",
            "affected_sensors": [],
            "confidence": "LOW",
            "confidence_reason": "LLM error"
        }, context

    try:
        raw_response = raw_response.strip()
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1

        if start != -1 and end != 0:
            json_str = raw_response[start:end]
            explanation = json.loads(json_str)
            logger.info("Response parsed successfully")
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        explanation = {
            "cause": raw_response,
            "severity": "UNKNOWN",
            "severity_reason": "Could not parse response",
            "action": "Review raw explanation",
            "affected_sensors": [],
            "confidence": "LOW",
            "confidence_reason": "Parse error"
        }

    return explanation, context


if __name__ == "__main__":
    print("Testing LLM Explainer")
    print("=" * 50)

    normal_url = "https://raw.githubusercontent.com/waico/SKAB/master/data/anomaly-free/anomaly-free.csv"
    df_normal = pd.read_csv(normal_url, sep=';',
                            index_col='datetime', parse_dates=True)

    test_anomaly = pd.Series({
        'Pressure': -0.82,
        'Volume Flow RateRMS': 30.1,
        'Current': 0.87,
        'Temperature': 74.3
    }, name=pd.Timestamp('2020-03-09 10:45:23'))

    print("Testing with simulated anomaly:")
    print(test_anomaly)
    print("\nCalling Ollama... (10-30 seconds)")
    print("=" * 50)

    explanation, context = explain_anomaly(
        test_anomaly,
        df_normal,
        anomaly_duration_seconds=47
    )

    print("\nCONTEXT SENT TO LLM:")
    print(context)
    print("\nLLM EXPLANATION:")
    print(json.dumps(explanation, indent=2))