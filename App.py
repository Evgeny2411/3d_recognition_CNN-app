import streamlit as st
import trimesh
import plotly.graph_objects as go

from model import Model

st.set_page_config(page_title='What is this shape', page_icon=":triangle:", layout='wide')
st.title('What is this shape')

st.header('Upload 3D Object')

user_file = st.file_uploader(label = "Tap", type = 'obj')
@st.cache_data
def load_points(file):
    mesh = trimesh.load(file, file_type='obj')
    points = mesh.vertices
    return points

@st.cache_resource
def make_prediction(points):
    classifier = Model()
    prediction_df = classifier.prediction(points = points)
    return prediction_df

@st.cache_resource
def visualize(points):
    fig = go.Figure(data = go.Scatter3d(x=points[:, 0], y = points[:, 1], z = points[:, 2], mode = 'markers'))

    fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, title=''),
                                 yaxis=dict(showticklabels=False, title=''),
                                 zaxis=dict(showticklabels=False, title='')))

    fig.update_layout(height = 600, autosize=True)
    st.plotly_chart(fig, use_container_width=True)

def main():
    if user_file is not None:

        points = load_points(user_file)
        prediction = make_prediction(points)
        shape = prediction.sort_values(by = 'Probs', ascending=False).index[0]

        col1, col2 = st.columns(2)

        with col1:
            st.header('Your model')
            visualize(points)

        with col2:

            st.header('Model guess : ' + str(shape))
            st.bar_chart(data=prediction.sort_values(by = 'Probs', ascending=False), use_container_width=True, height=600)


if __name__ == '__main__':
    main()