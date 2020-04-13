workspace(name="findPeople")



#copying tensorflow_servings example of forking tensorflow externally
# in accordance to https://github.com/tensorflow/tensorflow/issues/6706 

load("//findPeople:repo.bzl", "tensorflow_http_archive")

tensorflow_http_archive(
	name = "org_tensorflow",
	 sha256 = "06c16235bc49694052b035ba146429e71db229058cd6b43c02dbaa25101d25d0",
    git_commit = "dab4e7a725b0c92d5bad8dca683ba055d4a66584",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)


# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load("//findPeople:workspace.bzl", "findPeople_workspace")

findPeople_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.15.0")