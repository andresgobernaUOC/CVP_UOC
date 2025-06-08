import argparse as argparse
from datetime import datetime
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import polars as pl

#################################### Input Data Source
#input_file = os.path.join(f'D0_Source.csv')
#input_file='../SourceData/D0_Source.csv'
#D0_Source=pl.read_csv(input_file)

#################################### Model Parameters
#periodGroup=2  #Number of periods in a rolling window
#userMin_periods_in_PG=2 #min number of observations in the subset to perform the regression analysis
#userMax_periods_in_PG=2 #max number of observations in the subset to perform the regression analysis
#scenarioNameBase="TS PG02"  #User Defined

# Argument parser setup
parser = argparse.ArgumentParser(description='Run the model with parameters.')
parser.add_argument('input_file', type=str, help='CSV File')
parser.add_argument('periodGroup', type=int, help='Number of periods in a rolling window')
parser.add_argument('userMin_periods_in_PG', type=int, help='Minimum number of observations in the subset')
parser.add_argument('userMax_periods_in_PG', type=int, help='Maximum number of observations in the subset')
parser.add_argument('scenarioNameBase', type=str, help='Scenario name')

args = parser.parse_args()

# Assign variables from arguments
input_file = args.input_file
periodGroup = args.periodGroup
userMin_periods_in_PG = args.userMin_periods_in_PG
userMax_periods_in_PG = args.userMax_periods_in_PG
scenarioNameBase = args.scenarioNameBase

D0_Source=pl.read_csv(input_file)

for var_name, var_value in [
    ('periodGroup', periodGroup),
    ('userMin_periods_in_PG', userMin_periods_in_PG),
    ('userMax_periods_in_PG', userMax_periods_in_PG)
]:
    if not (1 <= var_value <= 36):
        print(f"Error: {var_name} must be an integer between 1 and 36. Got {var_value}.")
        sys.exit(1)

#################################### Model Parameters Validation
now = datetime.now()
runtTimeVersion = now.strftime("%Y%m%d%H%M")
inputPeriodsN=D0_Source.select("periodID").unique().shape[0]
os.system('cls')
print("**********************************************")
print(f"Running {now}")
print("**********************************************")
print("")
print(f"input_file: {input_file}")
print(f"periodGroup: {periodGroup}")
print(f"userMin_periods_in_PG: {userMin_periods_in_PG}")
print(f"userMax_periods_in_PG: {userMax_periods_in_PG}")
print(f"scenarioNameBase: {scenarioNameBase}")
print(f"Total Rows {D0_Source.shape[0]}")
print(f"Total Periods {inputPeriodsN}")

if(periodGroup>inputPeriodsN):
    periodGroup=inputPeriodsN
    
if(userMin_periods_in_PG>periodGroup):
    userMin_periods_in_PG=periodGroup

if(userMin_periods_in_PG>userMax_periods_in_PG):
    userMax_periods_in_PG=userMin_periods_in_PG

if(userMax_periods_in_PG>periodGroup):
    userMax_periods_in_PG=periodGroup
print("")
print("**********************************************")
print("Validated Parameters")
print("**********************************************")
print("")
print(f"periodGroup: {periodGroup}")
print(f"userMin_periods_in_PG: {userMin_periods_in_PG}")
print(f"userMax_periods_in_PG: {userMax_periods_in_PG}")
print("")
print("**********************************************")
print("Version Control")
print("**********************************************")
print("")
input_file = 'D0_VersionControl.csv'

try:
    
    D0_VersionControl = pl.read_csv(input_file)
    
    D0_VersionControl = D0_VersionControl.with_columns(
        pl.arange(1, D0_VersionControl.height + 1).alias('scenarioID')
    )
    
    scenarioID_var = D0_VersionControl['scenarioID'].max()
    scenarioID = scenarioID_var + 1
except FileNotFoundError:
    
    D0_VersionControl = pl.DataFrame()
    scenarioID = 1  


scenarioName = f"{scenarioNameBase} / (ID:{scenarioID}) R{periodGroup}/m{userMin_periods_in_PG}/M{userMax_periods_in_PG}"

print(f"Scenario: {scenarioName}")
print("")
New_Scenario_Row = {
    "scenarioName": scenarioName,
    "scenarioRunDate": int(now.strftime("%Y%m%d")),
    "periodGroup": periodGroup,
    "runtTimeVersion": int(runtTimeVersion)
}

New_Scenario = pl.DataFrame({
    "scenarioID": [scenarioID],
    **{k: [v] for k, v in New_Scenario_Row.items()}
})

if D0_VersionControl.height == 0:
    # If DataFrame was empty (new file), just assign the new DataFrame
    D0_VersionControl = New_Scenario
else:
    # Append new row
    D0_VersionControl = D0_VersionControl.vstack(New_Scenario)

# Write back to CSV
D0_VersionControl.write_csv(input_file, separator=',')

#################################### Data Cleansing
print("")
print("**********************************************")
print("Data Cleansing")
print("**********************************************")
print("")
D0_Source = D0_Source.with_columns([
    pl.when((pl.col("costInput").is_null()) 
            | (pl.col("costInput") <= 0)
            ).then(pl.lit("C")).otherwise(pl.lit("")
                                          ).alias("flag_costInput"),
    pl.when((pl.col("costDriverInput").is_null()) 
            | (pl.col("costDriverInput") <= 0)).then(pl.lit("V")).otherwise(pl.lit("")).alias("flag_costDriverInput")
])

D0_Source = D0_Source.with_columns([
    (pl.col("flag_costInput") + pl.col("flag_costDriverInput"))
    .alias("FlagColumn")
]).with_columns(
    pl.when((pl.col("FlagColumn")==pl.lit(""))).then(pl.lit("Valid")).otherwise(pl.col("FlagColumn")).alias("FlagColumn"))

dataCleansingCount = D0_Source.group_by("FlagColumn").agg(pl.len())

for row in dataCleansingCount.rows():
    print(f"{row[0]}: {row[1]}")

D0_Source = D0_Source.drop([ "flag_costInput", "flag_costDriverInput"])
D1_DataCleansing=D0_Source.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID"))
D1_Corporate=D0_Source.filter(pl.col("FlagColumn")=="Valid")

print("")
print("**********************************************")
blankmsg = ""
msg2print = blankmsg + "Modeling"
len_msg = len(msg2print)
print(msg2print)

#################################### Identify unique combinations for obsGroupID
blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Unique IDS"
len_msg = len(msg2print)
print(msg2print)



D1_Corporate_obsGroupID=D1_Corporate.select(
    "costActivity",
    "DataGranularityL1",
    "DataGranularityL2",
    "DataGranularityL3",
    "DataGranularityL4").unique().with_row_index(name="obsGroupID")
D1_Corporate=D1_Corporate.join(D1_Corporate_obsGroupID, on=[
    "costActivity",
    "DataGranularityL1",
    "DataGranularityL2",
    "DataGranularityL3",
    "DataGranularityL4"],how='inner')

#Identify Observations per obsGroupID
D1_Corporate_with_ID = (
    D1_Corporate
      .sort(["obsGroupID", "periodID"])
      .with_columns(
          pl.int_range(1, pl.len() + 1)
            .over("obsGroupID")
            .alias("observationID")
      )
)
#Output file Disabled, Only to use for internal control
#D1_Corporate_with_ID.select(    "observationID",    "obsGroupID",    "periodID",    "costInput",    "costDriverInput",    "costActivity",    "DataGranularityL1",    "DataGranularityL2",    "DataGranularityL3",    "DataGranularityL4"    ).with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D1_Corporate_with_ID.csv",separator=',')

D2_modelInput=D1_Corporate_with_ID.select(
    "observationID",
    "obsGroupID",
    "costInput",
    "costDriverInput"    
    )
#Output file Disabled, Only to use for internal control
#D2_modelInput.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D2_modelInput.csv",separator=',')


#################################### RollingPG Generation
def RollingPG(numberOfPeriods):
    groups = []
    num_groups = len(PeriodList) - numberOfPeriods + 1
    PeriodList_pd=PeriodList.to_pandas()
    for i in range(num_groups):
        group = PeriodList_pd.iloc[i:i + numberOfPeriods].copy()  # Copy to avoid SettingWithCopyWarning
        
        group['PGID'] = i + 1  # Add the PGID column with the current group number
        groups.append(group)
    # Concatenate all groups into a single DataFrame
    
    combined_df = pd.concat(groups).reset_index(drop=True)
    
    # Create a rank column within each PGID
    combined_df['SubPeriodID'] = combined_df.groupby('PGID').cumcount() + 1
    combined_df['SubPeriodID'] = combined_df['SubPeriodID'].astype(int)
    combined_df_pl=pl.from_pandas(combined_df)
    combined_df_pl=combined_df_pl.with_columns(pl.col('SubPeriodID').cast(pl.Int64).alias('SubPeriodID'))
    combined_df_pl=combined_df_pl.unique(maintain_order=False)    
    return combined_df_pl

PeriodList=D2_modelInput.select("observationID").unique().sort(["observationID"])
userNumberOfPeriods=periodGroup

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Rolling Periods"
len_msg = len(msg2print)
print(msg2print)

periodGroups=RollingPG(userNumberOfPeriods).sort(["observationID","PGID","SubPeriodID"])

#Output file Disabled, Only to use for internal control
#periodGroups.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D2_periodGroups.csv",separator=',')

D3_modelInput=D2_modelInput.join(periodGroups,on="observationID",how="inner")

#Output file Disabled, Only to use for internal control
#D3_modelInput.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D3_modelInput.csv",separator=',')

#0.0s

#################################### createBool Mapping
def createBool(numberOfPeriods,min_periods_in_PG,max_periods_in_PG):
    
    column_names = [str(i) for i in range(1, numberOfPeriods + 1)]
    num_rows = 2 ** numberOfPeriods
    
    boolTable = pd.DataFrame(np.nan, index=range(num_rows), columns=column_names)
    boolTable['RegressionPGID'] = range(1, num_rows+1) 
    boolTable['Binary'] = boolTable['RegressionPGID'].apply(lambda x: format(x-1, f'0{numberOfPeriods}b'))
    for i in range(numberOfPeriods):
        boolTable[column_names[i]] = boolTable['Binary'].apply(lambda x: int(x[i]))
    
    boolTable['SumOfBits'] = boolTable['Binary'].apply(lambda x: sum(int(bit) for bit in x))
    
    combined_df = pd.DataFrame()
    num_groups = len(PeriodList) - numberOfPeriods + 1
    for i in range(1, num_groups + 1):
        temp_df = boolTable.copy()
        temp_df['PGID'] = i
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            
    # Transpose the values of the columns 1, 2, 3, ..., numberOfPeriods for each PGID
    melted_df = combined_df.melt(id_vars=['PGID', 'RegressionPGID', 'Binary', 'SumOfBits'], 
                                 value_vars=column_names, 
                                 var_name='SubPeriodID', 
                                 value_name='Value')
    
    melted_df['SubPeriodID'] = melted_df['SubPeriodID'].astype(int)
    filtered_boolTable = melted_df[(melted_df['SumOfBits'] >= min_periods_in_PG)  & (melted_df['Value'] == 1) &(melted_df['SumOfBits'] <= max_periods_in_PG) ]
    
    output_df=pl.from_pandas(filtered_boolTable)
    
    output_df=output_df.unique(maintain_order=False) 
    output_df=output_df.with_columns(pl.col('SubPeriodID').cast(pl.Int64).alias('SubPeriodID'))
    return output_df

#################################### Execute createBool Mapping

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Masking Tables"
len_msg = len(msg2print)
print(msg2print)

boolTableMask=createBool(userNumberOfPeriods,userMin_periods_in_PG,userMax_periods_in_PG).select(
    "PGID","RegressionPGID","Binary","SumOfBits","SubPeriodID").sort("PGID","RegressionPGID")


#Output file Disabled, Only to use for internal control
#boolTableMask.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).write_csv("D4_boolTableMask.csv",separator=',')

D3_modelInput=D3_modelInput.join(boolTableMask, on=["PGID","SubPeriodID"],how="inner").select(
    "observationID","obsGroupID","costInput","costDriverInput","PGID","SubPeriodID","RegressionPGID").sort(["PGID","SubPeriodID","RegressionPGID"])
D4_modelInput=D3_modelInput.with_columns((pl.col('obsGroupID').cast(pl.Utf8)+"_"+ pl.col('PGID').cast(pl.Utf8)+"_"+ pl.col('RegressionPGID').cast(pl.Utf8)).alias('RegressionID')).rename({"costDriverInput":"CostDriver","costInput":"Cost"})

#Output file Disabled, Only to use for internal control
#D4_modelInput.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D4_modelInput.csv",separator=',')
#################################### Vectorized function to solve regressions
def manual_regression_worker(SelectedRegression):
    X = SelectedRegression['X'].values
    Y = SelectedRegression['Y'].values
    n = len(X)
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    if denominator == 0:
        VarUnitCost = np.nan   
        FixCost = np.nan   
    else:
        VarUnitCost = numerator / denominator
        FixCost = Y_mean - VarUnitCost * X_mean
    
    return [SelectedRegression.name, FixCost]  # VarUnitCost as Slope, not currently in use


def produceSolution(Input_df):
    Input_df_pd=Input_df.to_pandas()
    Input_df_grp = Input_df_pd.groupby("RegressionID")
    solution = Input_df_grp.apply(manual_regression_worker).tolist()
    solution_df = pd.DataFrame(solution, columns=['RegressionID', 'FixCost']) # VarUnitCost as Slope, not currently in use
    solution_df_pl=pl.from_pandas(solution_df)
    return solution_df_pl

#################################### Model Execution
max_cost_df = D4_modelInput.group_by('obsGroupID','PGID').agg(pl.col('Cost').min())
max_cost_df = max_cost_df.rename({'Cost': 'PGminCost'})
RegressionInput=D4_modelInput.select('RegressionID','CostDriver','Cost').rename({'CostDriver':'X','Cost':'Y'})

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Regressions"
len_msg = len(msg2print)
print(msg2print)

D5_modelOutput=produceSolution(RegressionInput).select("RegressionID","FixCost").join(D4_modelInput, on="RegressionID", how="inner" ).join(max_cost_df,on=("PGID","obsGroupID"),how="inner")

#Output file Disabled, Only to use for internal control
#D5_modelOutput.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D5_modelOutput.csv",separator=',')

#3.6s RP6
#s2m54s RP9  40.13M Regressions min model Acc 7.34 % AVG 89.56%
#>27m  RP12  544.78M Regressions min 14.74 AVG 89.26%
#################################### Business Rules
#Business Rule 1
blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Business rules"
len_msg = len(msg2print)
print(msg2print)

D6_modelOutput_BusinessRules=D5_modelOutput.filter(
    pl.col('FixCost')>=0) 
#Business Rule 2
D6_modelOutput_BusinessRules=D6_modelOutput_BusinessRules.filter(
    pl.col('FixCost')<pl.col('PGminCost')) 

#Output file Disabled, Only to use for internal control
#D6_modelOutput_BusinessRules.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv("D6_modelOutput_BusinessRules.csv",separator=',')


#################################### DEF Median Calculation
def calculate_median(D6_modelOutput):
    D6_modelOutput=D6_modelOutput.with_columns((pl.col('obsGroupID').cast(pl.Utf8)+"_"+ pl.col('PGID').cast(pl.Utf8)).alias('obsGroupIDPG_ID'))
    Solution_Median_step1 = D6_modelOutput.select(pl.col('obsGroupIDPG_ID','FixCost').sort_by('obsGroupIDPG_ID','FixCost'))
    Solution_Median_step2=Solution_Median_step1.group_by('obsGroupIDPG_ID', maintain_order=True).median()
        
    
    IQR=D6_modelOutput.group_by("obsGroupIDPG_ID").agg([
        pl.col("FixCost").quantile(0.25).alias("Q1"),
        pl.col("FixCost").quantile(0.75).alias("Q3"),
        pl.col("FixCost").median().alias("Median")
    ]).with_columns(
        ((pl.col("Q3") - pl.col("Q1")) / pl.col("Median")).alias("IQR")
    ).select(["obsGroupIDPG_ID", "IQR"])  
        
        
        
    Solution_Median_step2=Solution_Median_step2.rename({'FixCost':'FixCost_Solution'})
    Solution_Median_step3=Solution_Median_step1.group_by('obsGroupIDPG_ID').min()
    Solution_Median_step3=Solution_Median_step3.rename({'FixCost':'MinFixCost'})
    Solution_Median_step4=Solution_Median_step1.group_by('obsGroupIDPG_ID').max()
    Solution_Median_step4=Solution_Median_step4.rename({'FixCost':'MaxFixCost'})
    
    Solution_Median_step1a=Solution_Median_step1.join(Solution_Median_step2,left_on=['obsGroupIDPG_ID'], right_on=['obsGroupIDPG_ID'], how='inner')
    Solution_Median_step2a=Solution_Median_step1a.join(Solution_Median_step3,left_on=['obsGroupIDPG_ID'], right_on=['obsGroupIDPG_ID'], how='inner')
    Solution_Median_step3a=Solution_Median_step2a.join(Solution_Median_step4,left_on=['obsGroupIDPG_ID'], right_on=['obsGroupIDPG_ID'], how='inner')
    
    Solution_Median_step3a=Solution_Median_step3a.with_columns((pl.col('MinFixCost')-pl.col('FixCost_Solution')).alias('MinFixCost'))
    Solution_Median_step3a=Solution_Median_step3a.with_columns((pl.col('MaxFixCost')-pl.col('FixCost_Solution')).alias('MaxFixCost'))
    Solution_Median_step3a=Solution_Median_step3a.with_columns((pl.col('FixCost')-pl.col('FixCost_Solution')).alias('FixCost'))
    
    
    Solution_Median_step3a=Solution_Median_step3a.with_columns((-pl.col('FixCost')/pl.col('MinFixCost')).round(3).alias('MinFixCost'))
    Solution_Median_step3a=Solution_Median_step3a.with_columns((pl.col('FixCost')/pl.col('MaxFixCost')).round(3).alias('MaxFixCost'))
    
    Solution_Median_step3a=Solution_Median_step3a.with_columns(pl.col('MinFixCost').fill_nan(0))
    Solution_Median_step3a=Solution_Median_step3a.with_columns(pl.col('MaxFixCost').fill_nan(0))
    Solution_Median_step3a=Solution_Median_step3a.with_columns(pl.when(pl.col('FixCost') >= 0).then(pl.col('MaxFixCost')).otherwise(pl.col('MinFixCost')).alias('DispersionREF'))
        

    
    Solution_Median_step3a=Solution_Median_step3a.with_columns(
        pl.when((pl.col('DispersionREF') >= 0)&(pl.col('DispersionREF') <= 0.1)).then(pl.lit('P0')).
        when((pl.col('DispersionREF') > 0.1)&(pl.col('DispersionREF') <= 0.2)).then(pl.lit('P10')).
        when((pl.col('DispersionREF') > 0.2)&(pl.col('DispersionREF') <= 0.3)).then(pl.lit('P20')).
        when((pl.col('DispersionREF') > 0.3)&(pl.col('DispersionREF') <= 0.4)).then(pl.lit('P30')).
        when((pl.col('DispersionREF') > 0.4)&(pl.col('DispersionREF') <= 0.5)).then(pl.lit('P40')).
        when((pl.col('DispersionREF') > 0.5)&(pl.col('DispersionREF') <= 0.6)).then(pl.lit('P50')).
        when((pl.col('DispersionREF') > 0.6)&(pl.col('DispersionREF') <= 0.7)).then(pl.lit('P60')).
        when((pl.col('DispersionREF') > 0.7)&(pl.col('DispersionREF') <= 0.8)).then(pl.lit('P70')).
        when((pl.col('DispersionREF') > 0.8)&(pl.col('DispersionREF') <= 0.9)).then(pl.lit('P80')).
        when((pl.col('DispersionREF') > 0.9)).then(pl.lit('P90')).
        when((pl.col('DispersionREF') < 0)&(pl.col('DispersionREF') >= -0.1)).then(pl.lit('N0')).
        when((pl.col('DispersionREF') < -0.1)&(pl.col('DispersionREF') >= -0.2)).then(pl.lit('N10')).
        when((pl.col('DispersionREF') < -0.2)&(pl.col('DispersionREF') >= -0.3)).then(pl.lit('N20')).
        when((pl.col('DispersionREF') < -0.3)&(pl.col('DispersionREF') >= -0.4)).then(pl.lit('N30')).
        when((pl.col('DispersionREF') < -0.4)&(pl.col('DispersionREF') >= -0.5)).then(pl.lit('N40')).
        when((pl.col('DispersionREF') < -0.5)&(pl.col('DispersionREF') >= -0.6)).then(pl.lit('N50')).
        when((pl.col('DispersionREF') < -0.6)&(pl.col('DispersionREF') >= -0.7)).then(pl.lit('N60')).
        when((pl.col('DispersionREF') < -0.7)&(pl.col('DispersionREF') >= -0.8)).then(pl.lit('N70')).
        when((pl.col('DispersionREF') < -0.8)&(pl.col('DispersionREF') >= -0.9)).then(pl.lit('N80')).
        when((pl.col('DispersionREF') < -0.9)).then(pl.lit('N90')).
        otherwise(pl.lit('-666')).alias('DispersionCAT')
        )
    
    Solution_Median_W_Dis=Solution_Median_step3a.group_by('obsGroupIDPG_ID','DispersionCAT','FixCost_Solution').agg( pl.len().alias('NumberOfSolutions'))
    
    Solution_Median_W_Dis_pv=Solution_Median_W_Dis.pivot(values='NumberOfSolutions',index=['obsGroupIDPG_ID','FixCost_Solution'],on='DispersionCAT')
    Solution_Median_W_Dis_pv=Solution_Median_W_Dis_pv.fill_null(0)
    columns_to_check = ['-666','N0', 'N10', 'N20', 'N30', 'N40', 'N50', 'N60', 'N70', 'N80', 'N90','P0',
                    'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90']
    
    
    
    
    
    for col in columns_to_check:
        if col not in Solution_Median_W_Dis_pv.columns:
            Solution_Median_W_Dis_pv = Solution_Median_W_Dis_pv.with_columns(pl.lit(0).alias(col))
    
    
    review_solutions_step2=D6_modelOutput.select('observationID','obsGroupID','PGID','CostDriver','Cost','obsGroupIDPG_ID').unique()
    
    review_solutions_step9=review_solutions_step2.join(Solution_Median_W_Dis_pv, left_on=['obsGroupIDPG_ID'],right_on=['obsGroupIDPG_ID'], how='inner')
    review_solutions_step9=review_solutions_step9.join(IQR, left_on=['obsGroupIDPG_ID'],right_on=['obsGroupIDPG_ID'], how='inner')
    
    review_solutions_step9a=review_solutions_step9.select('observationID','obsGroupID','CostDriver','Cost','FixCost_Solution','P90',	'N90',	'N50',	'P10',	'N40',	'P60',	'P50',	'N20',	'P80',	'P70',	'N30',	'N70',	'N10',	'N60',	'P0',	'N0',	'P20',	'P30',	'N80',	'P40',	'-666',"IQR")
    review_solutions_step10=review_solutions_step9a.group_by('observationID','obsGroupID','CostDriver','Cost').agg([
        pl.col('FixCost_Solution').mean(),
        pl.col('IQR').mean().alias("AVG_IQR"),
        pl.col('P90','N90',	'N50',	'P10',	'N40',	'P60',	'P50',	'N20',	'P80',	'P70',	'N30',	'N70',	'N10',	'N60',	'P0',	'N0',	'P20',	'P30',	'N80',	'P40',	'-666').sum()
        ])
    
    
    Final_AVG=review_solutions_step10.with_columns((pl.col('Cost')-pl.col('FixCost_Solution')).alias('varCost'))
    
    
    
    Final_AVG=Final_AVG.select('observationID','obsGroupID','CostDriver','Cost','FixCost_Solution','varCost','N90','N80','N70','N60','N50','N40','N30','N20','N10','N0','P0','P10','P20','P30','P40','P50','P60','P70','P80','P90','-666',"AVG_IQR")
    return Final_AVG.with_columns(
    pl.sum_horizontal(pl.col([
        'N90','N80','N70','N60','N50','N40','N30','N20','N10','N0',
        'P0','P10','P20','P30','P40','P50','P60','P70','P80','P90','-666'
    ])).alias('TotalRegressions')
)
    

#################################### Median Calculation

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Median Value"
len_msg = len(msg2print)
print(msg2print)

D7_Avg_Median=calculate_median(D6_modelOutput_BusinessRules)



#output_folder = f""
#output_file = os.path.join(output_folder, f'D7_Avg_Median.csv')
#Output file Disabled, Only to use for internal control
#D7_Avg_Median.with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID")).write_csv(output_file,separator=',')
    

#################################### FILLING GAPS

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_Filling Missing Values"
len_msg = len(msg2print)
print(msg2print)

pre_modelOutput=D1_Corporate_with_ID.select("observationID","obsGroupID","costInput","costDriverInput").join(
    D7_Avg_Median.rename({
    "Cost":"costModel",
    "CostDriver":"costDriverModel",
    "FixCost_Solution":"fixCostModel",
})
    ,
        on=["observationID","obsGroupID"],
        how="left" ).with_columns(
    pl.when(
        (pl.col("fixCostModel") <= 0) | (pl.col("fixCostModel").is_null())
    )
    .then(pl.lit(-1))
    .otherwise(pl.lit(0))
    .alias("isValid"))

modelOutput_with_nulls=pre_modelOutput.select("obsGroupID","observationID","fixCostModel").sort("obsGroupID","observationID").pivot(values="fixCostModel",index="obsGroupID",on="observationID")

cols = [str(i) for i in range(1, modelOutput_with_nulls.width)]  


data = modelOutput_with_nulls.select(cols).to_numpy()


for r in range(data.shape[0]):
    row = data[r]
    not_null_indices = np.where(~np.isnan(row))[0]

    for i in range(len(not_null_indices) - 1):
        left_idx = not_null_indices[i]
        right_idx = not_null_indices[i + 1]

        gap = right_idx - left_idx
        if gap > 1:
            left_val = row[left_idx]
            right_val = row[right_idx]

            for j in range(1, gap):
                interp_val = left_val + (right_val - left_val) * j / gap
                row[left_idx + j] = interp_val

    data[r] = row


def ffill_row(row):
    for i in range(1, len(row)):
        if np.isnan(row[i]):
            row[i] = row[i-1]
    return row


def bfill_row(row):
    for i in range(len(row)-2, -1, -1):
        if np.isnan(row[i]):
            row[i] = row[i+1]
    return row


for r in range(data.shape[0]):
    data[r] = ffill_row(data[r])


for r in range(data.shape[0]):
    data[r] = bfill_row(data[r])


data_dict = {col: data[:, i] for i, col in enumerate(cols)}


df_filled = pl.DataFrame(data_dict)


modelOutput_filled = modelOutput_with_nulls.select("obsGroupID").with_columns(df_filled)

cols_except_obsGroupID = [col for col in modelOutput_filled.columns if col != "obsGroupID"]

modelOutput_filled=modelOutput_filled.unpivot(
    index=["obsGroupID"],
    on=cols_except_obsGroupID,
    variable_name="observationID",
    value_name="fixCostModel_FF"
).with_columns(
    pl.col("observationID").cast(pl.Int64)
)


modelOutput=pre_modelOutput.join(modelOutput_filled,on=["observationID","obsGroupID"],
        how="left" ).with_columns(pl.when(
                pl.col("fixCostModel").is_null()).then(pl.col("fixCostModel_FF")).otherwise(pl.col("fixCostModel")).alias("fixCostModel")).drop("fixCostModel_FF")


#modelOutput=modelOutput.with_columns(pl.when(pl.col("costInput")<=pl.col("fixCostModel")).then(pl.col("costInput")).otherwise(pl.col("fixCostModel")).alias("fixCost"))
modelOutput=modelOutput.with_columns(pl.col("fixCostModel").alias("fixCost"))

modelOutput=modelOutput.with_columns(
                                       (pl.col("costInput")-pl.col("fixCost")
                                        ).alias("varCost"))

modelOutput=modelOutput.with_columns(
    pl.arange(1, modelOutput.height + 1).alias('solutionID')
)

D7_Avg_Median=modelOutput.rename({"costModel":"Cost",
    "costDriverModel":"CostDriver",
    "fixCostModel":"FixCost_Solution"}).select(
    'observationID',	'obsGroupID',	'CostDriver',	'Cost',	'FixCost_Solution',	'varCost',	'N90',	'N80',	'N70',	'N60',	'N50',	'N40',	'N30',	'N20',	'N10',	'N0',	'P0',	'P10',	'P20',	'P30',	'P40',	'P50',	'P60',	'P70',	'P80',	'P90',	'-666',	'AVG_IQR',	'TotalRegressions','solutionID'
    )

modelOutput=modelOutput.with_columns(pl.lit(scenarioID).alias("scenarioID")).select("observationID","scenarioID","solutionID","isValid","costInput","costDriverInput","costModel","costDriverModel","fixCostModel","fixCost","varCost","obsGroupID").with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion"))

blankmsg = ""
for _ in range(len_msg):
    blankmsg = " " + blankmsg

msg2print = blankmsg + "|_ Modeling DONE"
len_msg = len(msg2print)
print(msg2print)

#################################### Preparation for Power BI Data ingestion
dataGranularity=D1_Corporate_with_ID.with_columns(pl.lit("Y").alias("isGranular")).with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).rename(    
    {"DataGranularityL1":"dataGranularityL1",
     "DataGranularityL2":"dataGranularityL2",
     "DataGranularityL3":"dataGranularityL3",
     "DataGranularityL4":"dataGranularityL4"}).select("isGranular","dataGranularityL1","dataGranularityL2","dataGranularityL3","dataGranularityL4","runtTimeVersion","observationID","obsGroupID")

modelInput=D1_Corporate_with_ID.with_columns(pl.lit("Y").alias("isGranular")).with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).rename(    
    {"DataGranularityL1":"dataGranularityL1",
     "DataGranularityL2":"dataGranularityL2",
     "DataGranularityL3":"dataGranularityL3",
     "DataGranularityL4":"dataGranularityL4"}).select(
         "observationID","obsGroupID","periodID","costInput","costDriverInput","costActivity",
                                                      "dataGranularityL1","dataGranularityL2","dataGranularityL3","dataGranularityL4","isGranular","runtTimeVersion")
modelscenario=D0_VersionControl.with_columns(
    pl.lit(userMin_periods_in_PG).alias("Min_Periods"),
    pl.lit(userMax_periods_in_PG).alias("Max_Periods")
    )
modelAccuracyOutput=D7_Avg_Median.select("obsGroupID",'N90','N80','N70','N60','N50','N40','N30','N20','N10','N0','P0','P10','P20','P30','P40','P50','P60','P70','P80','P90','-666',"solutionID","AVG_IQR").with_columns(pl.lit(runtTimeVersion).alias("runtTimeVersion")).with_columns(pl.lit(scenarioID).alias("scenarioID"))
modelAccuracyOutput=modelAccuracyOutput.unpivot(
    index=["obsGroupID","solutionID", "runtTimeVersion", "scenarioID","AVG_IQR"],
    on=[
        'N90','N80','N70','N60','N50','N40','N30','N20','N10','N0',
        'P0','P10','P20','P30','P40','P50','P60','P70','P80','P90',
        '-666'
    ],
    variable_name="dispersionCat",
    value_name="Value"
)

modelAccuracyOutput=modelAccuracyOutput.with_columns(pl.col("dispersionCat").str.replace("N0", "P0").alias("dispersionCat"))
modelAccuracyOutput=modelAccuracyOutput.with_columns(
    pl.when(pl.col("dispersionCat")=="P0").then(pl.lit("Med"))
    .when(pl.col("dispersionCat").str.starts_with("N")).then(pl.lit("-"))
    .when(pl.col("dispersionCat").str.starts_with("P")).then(pl.lit("+"))
    .otherwise(pl.lit(""))
    .alias("dispersionSig")
)

modelAccuracyOutput=modelAccuracyOutput.with_columns(pl.col("dispersionCat").str.replace("N", "-").alias("dispersionCat"))
modelAccuracyOutput=modelAccuracyOutput.with_columns(pl.col("dispersionCat").str.replace("P", "").alias("dispersionCat"))

modelAccuracyOutput=modelAccuracyOutput.with_columns(
pl.when(pl.col("dispersionSig")=="-").then(pl.lit("3"))
    .when(pl.col("dispersionSig")=="+").then(pl.lit("1"))
    .otherwise(pl.lit("2")).alias("dispersionSigOrder")
    
)





cols_to_keep = ["obsGroupID", "fixCost","runtTimeVersion","scenarioID"]

Intercept_modelOutput = modelOutput.with_columns([
    pl.col(col).alias(col) if col in cols_to_keep else pl.lit(0).alias(col)
    for col in modelOutput.columns
])

group_cols = [col for col in Intercept_modelOutput.columns if col != "fixCost"]

Intercept_modelOutput = Intercept_modelOutput.group_by(group_cols).agg(
    pl.col("fixCost").mean().alias("fixCost")
)

schema_modelOutput = modelOutput.schema

for col, dtype in schema_modelOutput.items():
    if col in Intercept_modelOutput.columns:
        Intercept_modelOutput = Intercept_modelOutput.with_columns(pl.col(col).cast(dtype))

Intercept_modelOutput = Intercept_modelOutput.with_columns(
    pl.when(pl.col("observationID") == 0)
      .then(pl.col("fixCost"))
      .otherwise(pl.col("costInput"))
      .alias("costInput")
)

Intercept_modelOutput=Intercept_modelOutput.select(
    'observationID','scenarioID','solutionID','isValid','costInput','costDriverInput','costModel',
'costDriverModel','fixCostModel','fixCost','varCost','obsGroupID','runtTimeVersion')

modelOutput_W_intercept = pl.concat([modelOutput, Intercept_modelOutput])

cols_to_keep = ["obsGroupID", "costInput","runtTimeVersion"
                
                ,"costActivity","dataGranularityL1","dataGranularityL2","dataGranularityL3","dataGranularityL4","isGranular"
                
                ]

Intercept_modelInput = modelInput.with_columns([
    pl.col(col).alias(col) if col in cols_to_keep else pl.lit(0).alias(col)
    for col in modelInput.columns
])



group_cols = [col for col in Intercept_modelInput.columns if col != "costInput"]

Intercept_modelInput = Intercept_modelInput.group_by(group_cols).agg(
    pl.col("costInput").mean().alias("costInput")
)

schema_modelInput = modelInput.schema

for col, dtype in schema_modelInput.items():
    if col in Intercept_modelInput.columns:
        Intercept_modelInput = Intercept_modelInput.with_columns(pl.col(col).cast(dtype))

Intercept_modelInput=Intercept_modelInput.select(
'observationID',
'obsGroupID',
'periodID',
'costInput',
'costDriverInput',
'costActivity',
'dataGranularityL1',
'dataGranularityL2',
'dataGranularityL3',
'dataGranularityL4',
'isGranular',
'runtTimeVersion'
)

modelInput_W_intercept = pl.concat([modelInput, Intercept_modelInput])

schema_modelInput = dataGranularity.schema
Intercept_dataGranularity=dataGranularity.with_columns(pl.lit(0).alias("observationID")).select(
    'isGranular',	'dataGranularityL1',	'dataGranularityL2',	'dataGranularityL3',	'dataGranularityL4',	'runtTimeVersion',	'observationID',	'obsGroupID'
).unique()
for col, dtype in schema_modelInput.items():
    if col in Intercept_dataGranularity.columns:
        Intercept_dataGranularity = Intercept_dataGranularity.with_columns(pl.col(col).cast(dtype))
dataGranularity_W_intercept=pl.concat([dataGranularity,Intercept_dataGranularity])

schema_modelInput = modelAccuracyOutput.schema

Intercept_modelAccuracyOutput=modelAccuracyOutput.with_columns(
    pl.lit("0").alias("solutionID"),
    pl.lit("-666").alias("dispersionCat"),
    pl.lit(0).alias("AVG_IQR"),
    pl.lit("").alias("dispersionSig"),
    pl.lit("2").alias("dispersionSigOrder"),
    pl.lit(0).alias("Value"),
    ).unique().select("obsGroupID",'solutionID',	'runtTimeVersion',	'scenarioID',	"AVG_IQR",'dispersionCat',	'Value',	'dispersionSig',	'dispersionSigOrder')

for col, dtype in schema_modelInput.items():
    if col in Intercept_modelAccuracyOutput.columns:
        Intercept_modelAccuracyOutput = Intercept_modelAccuracyOutput.with_columns(pl.col(col).cast(dtype))

modelAccuracyOutput_W_intercept=pl.concat([modelAccuracyOutput,Intercept_modelAccuracyOutput])

modelInput_W_intercept=modelInput_W_intercept.with_columns([
    pl.col("observationID").cast(pl.Utf8),
    pl.col("obsGroupID").cast(pl.Utf8),
    pl.concat_str(["observationID", "obsGroupID",pl.lit(scenarioID)], separator="_").alias("PK_ID")
])

modelOutput_W_intercept=modelOutput_W_intercept.with_columns([
    pl.col("observationID").cast(pl.Utf8),
    pl.col("obsGroupID").cast(pl.Utf8),
    pl.concat_str(["observationID", "obsGroupID",'scenarioID'], separator="_").alias("PK_ID")
]).with_columns([
    pl.col("obsGroupID").cast(pl.Utf8),
    pl.col("solutionID").cast(pl.Utf8),
    pl.concat_str(["obsGroupID", "solutionID",'scenarioID'], separator="_").alias("PK_IDs")
])

dataGranularity_W_intercept=dataGranularity_W_intercept.with_columns([
    pl.col("observationID").cast(pl.Utf8),
    pl.col("obsGroupID").cast(pl.Utf8),
    pl.concat_str(["observationID", "obsGroupID",pl.lit(scenarioID)], separator="_").alias("PK_ID")
])

modelAccuracyOutput_W_intercept=modelAccuracyOutput_W_intercept.with_columns([
    pl.col("obsGroupID").cast(pl.Utf8),
    pl.col("solutionID").cast(pl.Utf8),
    pl.concat_str(["obsGroupID", "solutionID",'scenarioID'], separator="_").alias("PK_IDs")
])

modelInput_W_intercept=modelInput_W_intercept.with_columns(pl.when(pl.col("observationID")=="0").then(pl.lit(" Modeled Cost")).otherwise(pl.lit("Business Input")).alias("Source"))
modelOutput_W_intercept=modelOutput_W_intercept.with_columns(pl.when(pl.col("observationID")=="0").then(pl.lit(" Modeled Cost")).otherwise(pl.lit("Business Input")).alias("Source"))
dataGranularity_W_intercept=dataGranularity_W_intercept.with_columns(pl.when(pl.col("observationID")=="0").then(pl.lit(" Modeled Cost")).otherwise(pl.lit("Business Input")).alias("Source"))
modelAccuracyOutput_W_intercept=modelAccuracyOutput_W_intercept.with_columns(pl.when(pl.col("solutionID")=="0").then(pl.lit(" Modeled Cost")).otherwise(pl.lit("Business Input")).alias("Source"))

#################################  APPEND MODEL RESULTS TO POWER BI DATA SOURCES 
output_folder = f"../Custom_PowerBIVisual/data/"


def cast_df_to_schema(df: pl.DataFrame, reference_df: pl.DataFrame) -> pl.DataFrame:
    ref_schema = reference_df.schema
    for col, dtype in ref_schema.items():
        if col in df.columns:
            df = df.with_columns(df[col].cast(dtype))
    return df

def append_polars_to_csv(df: pl.DataFrame, filename: str):
    output_file = os.path.join(output_folder, filename)
    if os.path.exists(output_file):
        existing_df = pl.read_csv(output_file)
        existing_df = cast_df_to_schema(existing_df, df)
        df = cast_df_to_schema(df, df)  # ensure df columns have correct types (optional)
        combined_df = pl.concat([existing_df, df])
    else:
        combined_df = df
    combined_df.write_csv(output_file)

output_file = os.path.join(output_folder, f'modelscenario.csv')
modelscenario.write_csv(output_file,separator=',')

append_polars_to_csv(dataGranularity_W_intercept, 'dataGranularity.csv')
append_polars_to_csv(modelInput_W_intercept, 'modelInput.csv')
append_polars_to_csv(modelAccuracyOutput_W_intercept, 'modelAccuracyOutput.csv')
append_polars_to_csv(modelOutput_W_intercept, 'modelOutput.csv')
append_polars_to_csv(D1_DataCleansing, 'DataCleansing.csv')
print("")
print("Files Appended to Power BI Report - Please Refresh Report")
print("")
print("Scenarios")
print(D0_VersionControl.select(pl.all()).to_pandas())
