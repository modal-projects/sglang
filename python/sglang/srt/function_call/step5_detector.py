"""Step-5 tool-call detector.

Step-5 uses the Step-3.5 / Qwen3-Coder XML format but, at greedy decoding, drops
the ``<`` before ``parameter=`` and puts the value inline after ``>``:

    <tool_call>
    <function=get_weather>
    parameter=city>San Francisco
    </parameter>
    </function>
    </tool_call>

Accept ``parameter=`` with or without the leading ``<``; everything else is the
Qwen3-Coder parser.
"""

import re

from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class Step5Detector(Qwen3CoderDetector):
    def __init__(self):
        super().__init__()
        self.parameter_prefix = "parameter="
        self.tool_call_parameter_regex = re.compile(
            r"<?parameter=(.*?)(?:</parameter>|(?=<?parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )
