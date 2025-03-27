# PII Masking Dataset Analysis Report

## 1. Missing Values Analysis

The dataset contains missing values in the following columns:
- source_text: 25 missing values (1250.00%)
- target_text: 25 missing values (1250.00%)

## 2. Text Length Analysis

Minimum text length: 36
Maximum text length: 512
Average text length: 420.98
Standard deviation: 57.00

## 3. PII Type Distribution
- TIME: 89225 instances
- USERNAME: 65405 instances
- IDCARD: 60014 instances
- SOCIALNUMBER: 59142 instances
- EMAIL: 58655 instances
- PASSPORT: 56759 instances
- DRIVERLICENSE: 55758 instances
- BOD: 54701 instances
- LASTNAME1: 54272 instances
- IP: 51724 instances
- GIVENNAME1: 48494 instances
- SEX: 47606 instances
- CITY: 46460 instances
- TEL: 46410 instances
- POSTCODE: 45316 instances
- STREET: 44645 instances
- STATE: 44557 instances
- BUILDING: 44454 instances
- TITLE: 40751 instances
- COUNTRY: 40040 instances
- DATE: 39735 instances
- PASS: 38296 instances
- SECADDRESS: 18542 instances
- LASTNAME2: 13634 instances
- GIVENNAME2: 12796 instances
- GEOCOORD: 4557 instances
- LASTNAME3: 4393 instances
- CARDISSUER: 50 instances

## 4. Token Distribution

Minimum tokens: 0
Maximum tokens: 491
Average tokens: 135.01

## 5. Special Characters Analysis

Total special characters: 8443130

Top 10 most common special characters:
- '"': 1410008 instances
- ':': 1356745 instances
- '-': 797008 instances
- '.': 714187 instances
- '*': 637001 instances
- ',': 607495 instances
- '<': 476808 instances
- '>': 476365 instances
- '/': 319789 instances
- '_': 238055 instances

## 6. Language Distribution
- French: 31447 instances (1572350.00%)
- German: 29976 instances (1498800.00%)
- English: 29908 instances (1495400.00%)
- Italian: 29066 instances (1453300.00%)
- Spanish: 28847 instances (1442350.00%)
- Dutch: 28433 instances (1421650.00%)

## 7. Label Consistency Analysis

Total samples: 177677
Consistent samples: 177677
Consistency percentage: 100.00%

## 8. BIO Label Distribution

Top 10 most common BIO labels:
- 'O': 18034320 instances
- 'I-IP': 760837 instances
- 'I-EMAIL': 517316 instances
- 'I-SOCIALNUMBER': 363532 instances
- 'I-DRIVERLICENSE': 334878 instances
- 'I-USERNAME': 315625 instances
- 'I-BOD': 279846 instances
- 'I-TEL': 276461 instances
- 'I-IDCARD': 227937 instances
- 'I-STREET': 215128 instances
