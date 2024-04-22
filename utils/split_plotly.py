import json
from constants import PLOTLY_START_FLAG, PLOTLY_END_FLAG


def split_plotly(
        output,
        start_flag=PLOTLY_START_FLAG,
        end_flag=PLOTLY_END_FLAG
):
    try:
        if PLOTLY_START_FLAG in output and PLOTLY_END_FLAG in output:
            start = output.find(start_flag) + len(start_flag)
            end = output.find(end_flag)
            region = output[start:end].strip()
            content = region[:output.find(start_flag)].strip()
            plotly_json = json.loads(region)
            return content, plotly_json
        else:
            return output, None

    except Exception as e:
        return output, None
