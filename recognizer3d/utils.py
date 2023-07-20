import json
import plotly.graph_objects as go
import streamlit as st
import trimesh

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

def load_dict(filepath: str) -> dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def load_points(filepath: str) -> np.ndarray:
    """Load a mesh from a file and return its vertices as a numpy array.

    Args:
        filepath (str): The path to the mesh file.

    Returns:
        np.ndarray: A numpy array containing the vertices of the mesh.
    """
    mesh = trimesh.load(filepath, file_type="obj")
    points = mesh.vertices
    return points


def visualize(points: np.ndarray) -> None:
    """Visualize a 3D scatter plot of the given points.

    Args:
        points (np.ndarray): A numpy array of shape (n, 3) representing the x, y, and z coordinates of n points.

    Returns:
        None
    """
    fig = go.Figure(
        data=go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers")
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        )
    )

    fig.update_layout(height=600, autosize=True)
    st.plotly_chart(fig, use_container_width=True)


def get_labels(encoder: LabelEncoder, labels: torch.Tensor):
    return encoder.inverse_transform(labels)