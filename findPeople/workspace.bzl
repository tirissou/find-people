# Again, following TfServings example for having tensorflow as an external dependency
# findPeople external dependencies that can be loaded in WORKSPACE
# files.

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def findPeople_workspace():
    """All TensorFlow Serving external dependencies."""

	tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

