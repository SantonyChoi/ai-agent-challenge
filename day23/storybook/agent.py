import json
import base64
from openai import OpenAI
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")

# --- Story Writer Agent ---
# Writes a 5-page children's story as structured data and stores it in state.

story_writer_agent = Agent(
    name="StoryWriter",
    description="Writes a 5-page children's story based on a theme.",
    instruction="""
You are a children's storybook writer. The user will give you a theme.

Write a 5-page children's story in Korean.

You MUST respond with ONLY a valid JSON array (no markdown, no backticks, no explanation).
Each element has:
- "page": page number (1-5)
- "text": the story text for that page (2-3 sentences, written for ages 4-8)
- "visual": a vivid visual description for an illustrator to draw (in English, detailed enough for image generation)

Example format:
[
  {"page": 1, "text": "옛날 옛적에...", "visual": "A small white rabbit standing in front of a mushroom house in a magical forest"},
  {"page": 2, "text": "...", "visual": "..."},
  ...
]

Rules:
- The story must have a clear beginning, middle, and happy ending.
- Use simple, warm language appropriate for young children.
- Visual descriptions must be detailed and whimsical, suitable for a picture book illustration style.
- Output ONLY the JSON array, nothing else.
""",
    output_key="story_pages",
    model=MODEL,
)


# --- Illustrator Agent ---
# Reads story pages from state and generates an image for each page.


async def generate_images(tool_context: ToolContext) -> dict:
    """Read story pages from state and generate an illustration for each page.
    Saves each image as an artifact."""
    story_pages_raw = tool_context.state.get("story_pages", "")

    # Parse the JSON from the story writer output
    try:
        pages = json.loads(story_pages_raw)
    except json.JSONDecodeError:
        # Try to extract JSON from possible markdown wrapping
        import re
        match = re.search(r'\[.*\]', story_pages_raw, re.DOTALL)
        if match:
            pages = json.loads(match.group())
        else:
            return {"error": "Could not parse story pages from state."}

    client = OpenAI()
    results = []

    for page in pages:
        page_num = page["page"]
        visual = page["visual"]
        text = page["text"]

        prompt = (
            f"Children's picture book illustration, soft watercolor style, "
            f"warm and friendly, for ages 4-8: {visual}"
        )

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
            )
            image_b64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)

            artifact = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            )
            filename = f"page_{page_num}.png"
            await tool_context.save_artifact(filename=filename, artifact=artifact)

            results.append({
                "page": page_num,
                "text": text,
                "visual": visual,
                "image_artifact": filename,
            })
        except Exception as e:
            results.append({
                "page": page_num,
                "text": text,
                "visual": visual,
                "image_error": str(e),
            })

    return {"pages": results}


illustrator_agent = Agent(
    name="Illustrator",
    description="Generates illustrations for each page of the story.",
    instruction="""
You are a children's book illustrator agent.

Your job: call the generate_images tool. It will read the story from state and create images for all 5 pages.

After the tool returns, present the results to the user in a nice format:

For each page, show:
- Page number
- Story text
- Visual description
- Whether the image was successfully generated (artifact filename)

Format example:
---
**Page 1**
Text: "옛날 옛적에..."
Visual: "A small rabbit..."
Image: page_1.png (saved as artifact)
---
""",
    tools=[generate_images],
    model=MODEL,
)


# --- Root Agent (Sequential) ---
# Runs Story Writer first, then Illustrator.

root_agent = SequentialAgent(
    name="StoryBookMaker",
    description="Creates a children's storybook with story and illustrations.",
    sub_agents=[story_writer_agent, illustrator_agent],
)
