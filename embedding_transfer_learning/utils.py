import re

def batchify(to_batch, batch_size):
    for i in range(0, len(to_batch), batch_size):
        yield to_batch[i:i + batch_size]



def make_recipe_string(recipe):
    name = recipe['name'].strip() if isinstance(recipe['name'], str) else "NOT GIVEN"
    description = recipe['description'].strip() if isinstance(recipe['description'], str) else "NOT GIVEN"
    ingredients = ", ".join(recipe['ingredients'])
    # "turn steps into numbered list"
    steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(recipe['steps'])])
    return f"""
DISH DESCRIPTION:
{name}

{description}

DISH INGREDIENTS:
{ingredients}

DISH STEPS:
{steps}"""


STEP_EXTRACTION_REGEX = r"^(?:\d+[\.-]\s?|\*+\s|-+\s|)"
ONELINE_STEPS_REGEX = r"(?:\d+\.\s?)(.*?)(?=\s\d+\.|\s*$)"


def extract_queries(queries_text: str):
    queries = queries_text.strip().split("\n")
    if len(queries) <= 2:
        single_row_text = " ".join(queries).strip()
        return [step.strip() for step in re.findall(ONELINE_STEPS_REGEX, single_row_text)]
    return [re.sub(STEP_EXTRACTION_REGEX, "", query.strip()).strip() for query in queries]


