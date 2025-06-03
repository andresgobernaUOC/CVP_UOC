**********************************************************************
Tables Description
    Custom_PowerBIVisual
        Contains both the source data and the source code for the Power BI report and its custom visual.
    Model
        Includes the Python script implementing the regression model used for analysis.
    SourceData
        Contains the synthetic data along with the script or method used to generate this data.

**********************************************************************

Cost-Volume-Profit (CVP) Modeling and Custom Power BI Visual
Overview
    This project implements a robust cost classification model using the Theil-Sen estimator to accurately separate 
    fixed and variable costs from volume data, including synthetic datasets with manufactured noise. It features a custom Power BI visual 
    designed to enhance cost-volume analysis by enabling visualization with three independent axes, offering new capabilities beyond existing marketplace visuals.

Features
    Robust fixed cost identification with high accuracy, even under noisy data conditions
    Custom Power BI visual with three independent axes to integrate more information per report


**********************************************************************

Getting Started
Run the Python scripts in model/ to generate cost classification results from your data.
Import the custom visual from the visual/ folder into Power BI.
Connect your data model and configure the visual to analyze fixed and variable costs against volume.

**********************************************************************
Contact
For questions or contributions, please open an issue or contact andresgoberna@uoc.edu
