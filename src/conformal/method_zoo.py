from conformal.classes.conformalizers import OTCPGlobalPredictor, OTCPLocalPredictor, SplitConformalPredictor
from conformal.classes.method_desc import ConformalMethodDescription


section5 = [
    ConformalMethodDescription(
        name="PB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{pb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Rank",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
    ),
    ConformalMethodDescription(
        name="RPB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{rpb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Quantile",
        class_name="OTCPGlobalPredictor",
        cls=OTCPGlobalPredictor
    ),
    ConformalMethodDescription(
        name="HPD",
        name_mathtext=r"$\mathcal{C}^{\mathrm{HPD}}$",
        base_model_name="CVQRegressor",
        score_name="Log Density",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
        kwargs=dict(lower_is_better=False)
    ),

]


baselines = [
    ConformalMethodDescription(
        name="OT-CP-Global",
        name_mathtext=r"$\mathrm{OT}$-$\mathrm{CP}$",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="OTCPGlobalPredictor",
        cls=OTCPGlobalPredictor
    ),
    ConformalMethodDescription(
        name="OT-CP-Local",
        name_mathtext=r"$\mathrm{OT}$-$\mathrm{CP}$+",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="OTCPLocalPredictor",
        cls=OTCPLocalPredictor
    ),

]