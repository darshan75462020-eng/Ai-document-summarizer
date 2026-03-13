import sys
import os

# Add the 'app' directory to the path so we can import summarizer
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from summarizer import generate_ai_insights

long_text = "This is a dummy text. " * 2000 # 8000 words roughly, since 4 words per repetition.

print(f"Testing with text length: {len(long_text)} characters, ~{len(long_text.split())} words.")

insights = generate_ai_insights(long_text)

print("\n--- Final Summary ---")
print(insights['summary'])
print("\n--- Final Bullets ---")
print(insights['bullets'])
