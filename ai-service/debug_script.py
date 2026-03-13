import sys
import traceback

with open("error.log", "w", encoding="utf-8") as f:
    try:
        import test_script
    except Exception as e:
        traceback.print_exc(file=f)
