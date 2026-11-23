def score_layers(file_type: str, layers: dict) -> dict:
    reasons = []
    score = 0
    hash_result = layers.get("hash", {})
    bazaar_result = layers.get("malwarebazaar", {})
    yara_result = layers.get("yara", {})
    pe_result = layers.get("pe", {})
    ml_result = layers.get("ml", {})

    if hash_result.get("matched"):
        score = 100
        source = hash_result.get("source", "local-list")
        reasons.append(f"SHA-256 matched {source}")

    if bazaar_result.get("matched"):
        score = 100
        reasons.append("SHA-256 found in MalwareBazaar")

    for match in yara_result.get("matches", []):
        if match.get("meta", {}).get("confidence") == "high":
            score = 100
        else:
            score = max(score, 40)
        reasons.append(f"YARA rule matched: {match['rule']}")

    if ml_result.get("status") == "complete":
        probability = ml_result["probability"]
        threshold = ml_result["threshold"]
        if probability >= threshold:
            score = min(100, score + 50)
            model_name = {
                "ember_full_v2": "EMBER full v2 model",
                "byte_histogram_entropy_strings_v1": "EMBER byte, entropy and strings model",
            }.get(ml_result.get("feature_set"), "EMBER histogram model")
            reasons.append(f"{model_name} {probability:.3f} >= {threshold:.3f}")
        else:
            score = min(100, score + round(probability * 20))

    if pe_result.get("status") == "complete":
        for section in pe_result["sections"]:
            if section["entropy"] > 7.2 and section["raw_size"] >= 1024:
                score = min(100, score + 15)
                reasons.append(f"section {section['name']} entropy {section['entropy']} > 7.2")
        if pe_result["upx_sections"]:
            score = min(100, score + 20)
            reasons.append("UPX section names: " + ", ".join(pe_result["upx_sections"]))
        imports = pe_result["suspicious_imports"]
        if imports:
            score = min(100, score + min(30, len(imports) * 10))
            reasons.append("imports " + ", ".join(imports))
        if pe_result["timestamp_anomaly"]:
            score = min(100, score + 5)
            reasons.append("PE timestamp is absent or in the future")

    if file_type in {"zip", "unknown", "pdf", "elf", "png"} and score < 80:
        reason = (
            "Archive contents were not analyzed"
            if file_type == "zip"
            else "Only hash and YARA checks support this file type"
            if file_type in {"pdf", "elf", "png"}
            else "File type is unsupported"
        )
        return {"verdict": None, "score": None, "reasons": [reason]}
    if any(result.get("status") == "error" for result in layers.values()) and score < 80:
        return {"verdict": None, "score": None, "reasons": ["One or more analysis layers failed"]}
    verdict = "malicious" if score >= 80 else "suspicious" if score >= 30 else "clean"
    if not reasons:
        reasons.append("No indicators found in the completed checks")
    return {"verdict": verdict, "score": score, "reasons": reasons}
