import re

class GuardrailManager:
    def __init__(self):
        #patterns for PII
        self.email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        self.phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"

        #known prompt injection attacks sample
        self.injection_keywords = [
        "ignore previous instructions",
        "reveal system prompt",
        "system prompt",
        "ignore all rules",
        "override safety settings",
    ]

    # Pre-Guardrail
    def validate_input(self,query: str) -> tuple[bool,str,str]:
        #detects prompt injections and redacts PII

        query_lower = query.lower()

        #check for prompt injection
        for keyword in self.injection_keywords:
            if keyword in query_lower:
                return (False, query, f"Prompt injection detected: '{keyword}'")

        #redacts PII
        santized = re.sub(self.email_pattern,"[REDACTED_EMAIL]", query)
        sanitized = re.sub(self.phone_pattern, "[REDACTED_PHONE]", sanitized)

        return (True,sanitized,"input is safe..")

    # Post-Guardrail
    def validate_output(self, response: str,retrieved_contexts: list[str]
  ) -> tuple[bool, str]:
        # verifies the ans is grounded in retrived text

        if not retrieved_contexts:
            return (False, "No context available to verify response.")

        #extract the content words
        response_words = [
            w.lower()
            for w in response.split()
            if len(w) > 4 and w.isalnum()
        ]

        if not response_words:
            return (True, "Response verified..")

        # measure with percentages of response words that appear in the retrieved context..
        matches = sum(1 for word in response_words if word in combined_context)
        grounded_ratio = matches / len(response_words)

        # If less than 30% of content words exist in context, flag as hallucination
        if grounded_ratio < 0.30:
            return (
                False,
                "Response failed groundedness check (potential hallucination).",
            )

        return (True, "Response verified")

            
