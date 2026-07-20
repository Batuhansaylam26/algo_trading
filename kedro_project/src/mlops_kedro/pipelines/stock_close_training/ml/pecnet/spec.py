from __future__ import annotations

from .spec_class import *  # noqa: F403
from .spec_class import PecnetSpecBuilder

pecnet_spec_builder = PecnetSpecBuilder()
build_pecnet_spec = pecnet_spec_builder.build_pecnet_spec
to_pecnet_frame = pecnet_spec_builder.to_pecnet_frame
make_pecnet_train_test_split = pecnet_spec_builder.make_pecnet_train_test_split
