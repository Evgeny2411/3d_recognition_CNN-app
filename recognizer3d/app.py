from pathlib import Path
import streamlit as st

from config import config
from recognizer3d import utils
from recognizer3d.model import Model

st.set_page_config(page_title="What is this shape", page_icon=":triangle:", layout="wide")
st.title("What is this shape")
st.header("Upload 3D Object")
user_file = st.file_uploader(label="Tap", type="obj")


def main():
    args = Path(config.CONFIG_DIR, "args.json")
    model = Model(args)
    if user_file is not None:

        points = utils.load_points(user_file)
        prediction = model.make_prediction(points)
        shape = prediction.index.get(0, 'Unknown shape')

        col1, col2 = st.columns(2)

        with col1:
            st.header("Your model")
            utils.visualize(points)
        with col2:
            st.header("Model guess : " + shape)
            st.bar_chart(
                data=prediction.sort_values(by="Probs", ascending=False),
                use_container_width=True,
                height=600,
            )
    else:
        st.warning("Please select a file.")


if __name__ == "__main__":
    main()
