"""Generic statistical plot: pool one product, render via ``PlotPanel``.

A :class:`PoolPlot` is the common shape of every "tidy columns over a single
product" plot (shape descriptors, dynamics curves, …). It consumes exactly one
``quantity_id``, pools that product across the in-scope positions
(:func:`~cellflow.napari.aggregate_quantification.plots._pooling.pool_quantity`),
and hands the tidy frame to the quantity-agnostic
:class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel`.

Subclasses set ``plot_id`` / ``display_name`` / ``family``, the single
``consumes`` product, and the ``value_columns`` the panel offers. Set
``label_field`` to wire click-to-load against that catalogue path column.

The heavy read is :meth:`pool` (headless, off-thread-safe); :meth:`create_panel`
builds the Qt panel and must run on the GUI thread. The plot area may pool first
(off-thread) and pass the result into ``create_panel`` to stay responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from cellflow.aggregate_quantification.quantifier import Quantifier
from cellflow.napari.aggregate_quantification.plots import Plot, PlotContext, PlotParams
from cellflow.napari.aggregate_quantification.plots._pooling import (
    GROUP_COLUMNS,
    pool_quantity,
)


class PoolPlot(Plot):
    """A statistical plot over one pooled product."""

    #: Value columns the panel offers for the y-axis (filtered to those the
    #: pooled snapshot actually carries).
    value_columns: ClassVar[tuple[str, ...]] = ()
    #: Group-by axes the panel offers. Defaults to the catalogue metadata +
    #: ``class_label`` + ``frame``; views without a per-frame axis (per-track,
    #: per-tissue) narrow it.
    group_columns: ClassVar[tuple[str, ...]] = GROUP_COLUMNS
    #: Catalogue record field holding the label TIFF for click-to-load, or
    #: ``None`` to disable picking → viewer.
    label_field: ClassVar[str | None] = None
    #: Whether to left-join the NLS subpopulation ``class_label`` at pool time.
    #: Off for position-level views (per-tissue) that have no per-cell rows.
    join_class: ClassVar[bool] = True
    #: Plot type the panel opens on (``""`` = the panel's own default). Lets a
    #: view land directly on its natural rendering (``box`` / ``bar`` /
    #: ``potential``) instead of the generic first option.
    default_plot: ClassVar[str] = ""
    #: Open the panel with adaptive (variable-width) bins — used by the potential
    #: landscape so sparse tails do not read as spurious wells.
    default_adaptive_bins: ClassVar[bool] = False

    @property
    def quantity_id(self) -> str:
        """The single product this plot pools (pool plots consume exactly one)."""
        if len(self.consumes) != 1:
            raise ValueError(
                f"{type(self).__name__} must consume exactly one product, "
                f"got {self.consumes!r}"
            )
        return self.consumes[0]

    def _read_table(self, quantifier: Quantifier, path: Path) -> Any:
        """The per-position table to pool. Defaults to the product's tidy
        ``object_table``; override to pool a different view (e.g. per-track)."""
        return quantifier.object_table(path)

    def pool(self, records: list[dict]) -> pd.DataFrame:
        """Pool the consumed product across *records* (headless / off-thread)."""
        return pool_quantity(
            self.quantity_id,
            records,
            table_fn=self._read_table,
            join_class=self.join_class,
        )

    def prepare(self, records: list[dict], params: PlotParams = PlotParams()) -> pd.DataFrame:
        # The standard one-product pool ignores plot-time params; views that need
        # them (potential landscape, density, z-score) override prepare.
        return self.pool(records)

    def create_panel(self, ctx: PlotContext, prepared: pd.DataFrame | None = None) -> Any:
        """Build the ``PlotPanel`` bound to the pooled snapshot.

        *prepared* lets the plot area pass an off-thread pooling result; when
        ``None`` it pools inline.
        """
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
        from cellflow.napari.aggregate_quantification.plugins._click_to_load import (
            ClickToLoad,
        )

        df = self.prepare(ctx.records) if prepared is None else prepared
        # A fresh controller per panel always targets the current viewer and owns
        # every Load from this panel, so its "replace previous layer" guarantee
        # holds for the panel's lifetime.
        controller = ClickToLoad(ctx.viewer)
        resolver = (
            controller.resolver(ctx.records, self.label_field)
            if self.label_field
            else None
        )
        return PlotPanel(
            df,
            value_columns=self.value_columns,
            group_columns=self.group_columns,
            target_resolver=resolver,
            loader=controller.load,
            default_plot=self.default_plot,
            default_adaptive_bins=self.default_adaptive_bins,
        )
