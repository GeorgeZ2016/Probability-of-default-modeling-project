/*-----------------------------------------------
 *NAME: DATA CLEANING.sas

 *PURPOSE: Use the result from last homework to calculate the value of actual losses.

 
------------------------------------------------*/
/*Convert the servicing text (csv) ?le to a SAS dataset*/

data work.orig;
infile "E:\Graduate Study\career\SAS\hw6\sample 2000\sample_orig_2000.txt"
dlm="|" MISSOVER DSD lrecl=32767 firstobs=1;
input
fico : 8.
dt_first_pi : 8.
flag_fthb : $1.
dt_matr : 8.
cd_msa : 8.
mi_pct : 8.
cnt_units : 8.
occpy_sts : $1.
cltv : 8.
dti : 8.
orig_upb : 8.
ltv : 8.
int_rt : 8.
channel : $1.
ppmt_pnlty : $1.
prod_type : $5.
st : $2.
prop_type : $2.
zipcode : $5.
id_loan : $16.
loan_purpose : $5.
orig_loan_term : 8.
cnt_borr : $2.
seller_name : $20.
servicer_name : $20. ;
label 
fico='CREDIT SCORE'
dt_first_pi='FIRST PAYMENT DATE'
flag_fthb='FIRST TIME HOMEBUYER FLAG'
dt_matr='MATURITY DATE'
cd_msa='MSA'
mi_pct='MORTGAGE INSURANCE PERCENTAGE'
cnt_units='NUMBER OF UNITS'
occpy_sts='OCCUPANCY STATUS'
cltv='ORIGINAL COMBINED LOAN-TO-VALUE'
dti='ORIGINAL DEBT-TO-INCOME RATIO'
orig_upb='ORIGINAL UPB'
ltv='ORIGINAL LOAN-TO-VALUE'
int_rt='ORIGINAL INTEREST RATE'
channel='CHANNEL'
ppmt_pnlty='PREPAYMENTPENALTY MORTGAGE FLAG '
prod_type='PRODUCT TYPE'
st='PROPERTY STATE'
prop_type='PROPERTY TYPE'
zipcode='POSTAL CODE'
id_loan='LOAN SEQUENCE NUMBER'
loan_purpose='LOAN PURPOSE'
orig_loan_term='ORIGINAL LOAN TERM'
cnt_borr='NUMBER IF BORROWERS'
seller_name='SELLER NAME'
servicer_name='SERVICER NAME'
run;

data svcgfile; 
infile "E:\Graduate Study\career\SAS\hw6\sample 2000\sample_svcg_2000.txt" 
dlm="|" MISSOVER DSD lrecl=32767 firstobs=1;
	input 
		ID_loan          :  $12. 
		Period           :   8. 
		Act_endg_upb    :   8. 
		delq_sts         :   $8. 
		loan_age         :   8. 
		mths_remng       :   8. 
		repch_flag      :   $1. 
		flag_mod           :   $1. 
		CD_Zero_BAL     :   $3. 
		Dt_zero_BAL        :   8. 
		New_Int_rt         :   8.  
		Amt_Non_Int_Brng_Upb  : 12. 
		Dt_Lst_Pi  : 6. 
		MI_Recoveries : 12. 
		Net_Sale_Proceeds  : $14. 
		Non_MI_Recoveries  : 12. 
		Expenses : 12.  
		legal_costs  :  12. 
		maint_preserv_costs : 12. 
		Taxes_ins_costs :  12. 
		misc_costs  :  12. 
		actual_loss   : 12. ; 
/*define the label of variables*/
label 
	ID_loan='LOAN SEQUENCE NUMBER'
	Period='MONTHLY REPORTING PERIOD'
	Act_endg_upb='CURRENT AUTUAL UPB' 
	delq_sts='CURRENT LOAN DELIQUENCY STATUS'
	loan_age='LOAN AGE'
	mths_remng='REMAINING MONTHS TO LEGAL MATURITY' 
	repch_flag='REPURCHASE FLAG'
	flag_mod='MODIFICATION FLAG' 
	CD_Zero_BAL='ZERO BALANCE CODE'
	Dt_zero_BAL='ZERO BALANCE EFFECTIVE DATE'
	New_Int_rt='CURRENT INTEREST RATE'
	Amt_Non_Int_Brng_Upb='CURRENT DEFERRED UPB'
	Dt_Lst_Pi='DUE DATE OF LAST PAID INSTALLMENT'
	MI_Recoveries='MI RECOVERIES'
	Net_Sale_Proceeds='NET SALES PROCEEDS'
	Non_MI_Recoveries='NON MI RECOVERIES'
	Expenses='EXPENSES'
	legal_costs='LEGAL COSTS' 
	maint_preserv_costs='MAINTENANCE AND PRESERVATION COSTS'
	Taxes_ins_costs='TAXES AND INSURANCE'
	misc_costs='MISCELLANEOUS EXPENSES'
	actual_loss='ACTUAL LOSS CALCULATION';
run; 
/*create the format of variables*/
proc format;
	value $delinquency_status 
		'0'='Current'
		'1'='30-59 days delinquent'
		'2'='60-89 days delinquent'
		'3'='90-119 days delinquent'
		'4','5','6','7','8','9'='120+ days delinquent'
		'R'='REO acquisition'
		'XX'='Unknown'
		' '='Unavailable';
	value $repurchase_flag
		'N'='Not Repurchased'
		'Y'='Repurchased'
		' '='Not modified';
	value $modification_flag
		'Y'='Yes'
		' '='Not modified';
	value $zero_balance_code
		'01'='Prepaid or Matured'
		'03'='Foreclosure Alternative Group'
		'06'='Repurchase prior to Property Disposition'
		'09'='REO Disposition'
		' '='Not Applicable';
	value $zero_balance_effective_date
		' '='Not Applicable';
	value $net_sales_proceeds
		'C'='Covered'
		'U'='Unknown';
run;
/*Modify the SAS dataset to create work.friendlier, a dataset that has labels and formats that re?ect the meaning of the variables. */
data work.friendlier(drop=year month period);
	set svcgfile;
	format
		ID_loan            $12. 
		Period             6. 
		Act_endg_upb       12. 
		delq_sts           $delinquency_status. 
		loan_age           8. 
		mths_remng         8. 
		repch_flag         $repurchase_flag. 
		flag_mod           $modification_flag. 
		CD_Zero_BAL        $zero_balance_code. 
		Dt_zero_BAL        8.
		New_Int_rt         8.  
		Amt_Non_Int_Brng_Upb   12. 
		Dt_Lst_Pi          6. 
		MI_Recoveries      12. 
		Net_Sale_Proceeds  $net_sales_proceeds. 
		Non_MI_Recoveries  12. 
		Expenses           12.  
		legal_costs        12. 
		maint_preserv_costs    12. 
		Taxes_ins_costs    12. 
		misc_costs         12. 
		actual_loss        12. 
		Report_Month       MMYYS.
		zero_balance_effective_date    MMYYS.
		DUE_DATE    MMYYS.
; 
/*Convert the values in date variables to SAS dates, and display them using the format MMYYS. */
	/*reporting month*/
	Year = floor(period/100); 
	Month = period-100*Year; 
	Report_Month = mdy(Month,1,Year);
	/*zero balance effective date*/
	Year = floor(Dt_zero_BAL/100); 
	Month = Dt_zero_BAL-100*Year; 
	zero_balance_effective_date = mdy(Month,1,Year);
	/*DUE DATE OF LAST PAID INSTALLMENT*/
	Year = floor(Dt_Lst_Pi/100); 
	Month = Dt_Lst_Pi-100*Year; 
	DUE_DATE = mdy(Month,1,Year);
run;
/*Creat the actual losses dataset*/
data work.actual_losses(drop=Last_endg_upb Last_interest_rate Last_Loan_Age Last_Mths_Remng);
/*creat label to explain the variables*/
label Current_Interest_Rate='CURRENT INTEREST RATE'
	  EAD='EXPOSURE AT DEFAULT (EAD)'
	  Percent_Loss='ACTUAL LOSS CALCULATION/EAD(LGD)'
	  zero_balance_effective_date='EFFECTIVE DATE WHEN ZERO BALANCE'
	  Report_Month='REPORTED MONTH'
	  ;
/* if this is not the last observation for a loan, save the values of variables for imputing values */ 
retain Act_endg_upb 0
	   Current_Interest_Rate 0
       Loan_Age 0
	   Mths_Remng 0
	   ;
Last_endg_upb = Act_endg_upb; 
Last_interest_rate = Current_Interest_Rate; 
Last_Loan_Age = Loan_Age; 
Last_Mths_Remng = Mths_Remng;
	set work.friendlier;
	by ID_loan;
/*Find the data with actual_loss which is not a missing value.*/
if actual_loss ne . then do;
	if Last.ID_loan then do;
		Act_endg_upb = Last_endg_upb; 
		Current_Interest_Rate = Last_interest_rate; 
		Loan_Age = Last_Loan_Age; 
		Mths_Remng = Last_Mths_Remng;
	end;
/*calculate the EAD and Percent_loss according to the requirement.*/
EAD= Act_endg_upb+Taxes_ins_costs+Expenses+maint_preserv_costs+misc_costs;
Percent_Loss = actual_loss/EAD;
output;

end;
run;


data project
(keep=dti int_rt fico ltv loss_flag orig_upb cltv orig_loan_term fico_shifted);
merge work.orig
	  work.actual_losses(IN=in_loss);  
	  by id_loan;

fico_shifted=fico-300;

if in_loss then loss_flag=1;
else loss_flag=0;

run;


/* Use PROC CONTENTS on work.actual_losses. */
proc contents data=project;
	title 'Contents of the loan with actual losses';
run;
/*Show 20 observations corresponding to this loan using PROC PRINT.*/
proc print data=project(obs=20)label;
	title 'Observations of the loan with actual losses';
run;



proc export data=project outfile="E:\project.csv";
run;

proc corr data=project;
var loss_flag dti fico int_rt orig_upb ltv;
run;

