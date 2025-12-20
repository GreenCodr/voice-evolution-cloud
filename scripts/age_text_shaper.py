def shape_text_for_age(text: str, age: int) -> str:
    if age <= 8:
        return (
            "Hi! "
            + text.replace(".", "! ")
            + " I like talking. "
        )

    if age >= 70:
        return (
            text
            + " ... I have lived a long life."
        )

    return text