Cost-Volume-Profit (CVP) Modeling and Custom Power BI Visual
Overview
This project implements a robust cost classification model using the Theil-Sen estimator to accurately separate fixed and variable costs from volume data, including synthetic datasets with manufactured noise. It features a custom Power BI visual designed to enhance cost-volume analysis by enabling visualization with three independent axes, offering new capabilities beyond existing marketplace visuals.

Features
Robust fixed cost identification with high accuracy, even under noisy data conditions

Detection and mapping of outliers for improved future iterations

Custom Power BI visual with three independent axes to integrate more information per report

Performance optimization by reducing the number of visuals per report page

Use cases demonstrating strategic business impacts including economy of scale analysis, pricing modeling, and operational efficiency benchmarking

Repository Contents
model/ — Python scripts implementing the Theil-Sen regression and synthetic data generation

visual/ — Source code for the custom Power BI visual using TypeScript and Power BI SDK

data/ — Sample synthetic datasets with varying noise scenarios

docs/ — Project documentation including report chapters, glossary, and use case analysis

Getting Started
Run the Python scripts in model/ to generate cost classification results from your data.

Import the custom visual from the visual/ folder into Power BI.

Connect your data model and configure the visual to analyze fixed and variable costs against volume.

Future Work
Migrate model execution and data hosting to cloud-based relational databases

Enhance visual customization options (color, fonts, settings)

Develop outlier mapping capabilities integrated within the visual

Contact
For questions or contributions, please open an issue or contact andresgoberna@uoc.edu
