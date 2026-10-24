import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        co = frame.f_code
        func_name = co.co_name
        func_line_no = frame.f_lineno
        func_filename = co.co_filename
        if "transformers" in func_filename:
            print(f"Call {func_name} at {func_filename}:{func_line_no}", flush=True)
    return trace_calls

sys.settrace(trace_calls)
print("Tracing import transformers...", flush=True)
import transformers
print("Imported transformers successfully!", flush=True)
