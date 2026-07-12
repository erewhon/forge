"""Ecosystem adapters: the per-package-manager port behind the bumper's neutral loop."""

from forge.dependabot.ecosystems.base import (
    Ecosystem as Ecosystem,
)
from forge.dependabot.ecosystems.base import (
    EcosystemError as EcosystemError,
)
from forge.dependabot.ecosystems.base import (
    detect_ecosystem as detect_ecosystem,
)
from forge.dependabot.ecosystems.base import (
    present_ecosystems as present_ecosystems,
)
from forge.dependabot.ecosystems.base import (
    resolve_ecosystem as resolve_ecosystem,
)
