import json
from constants import PLOTLY_START_FLAG, PLOTLY_END_FLAG


def split_plotly(
        output,
        start_flag=PLOTLY_START_FLAG,
        end_flag=PLOTLY_END_FLAG
):
    try:
        start = output.find(start_flag) + len(start_flag)
        end = output.find(end_flag)
        region = output[start:end].strip()

        content = region[:output.find(start_flag)].strip()
        # st.markdown(content)

        plotly_json = json.loads(region)
        # st.plotly_chart(plotly_json, use_container_width=True)

        return content, plotly_json
    except Exception as e:
        return "", ""
