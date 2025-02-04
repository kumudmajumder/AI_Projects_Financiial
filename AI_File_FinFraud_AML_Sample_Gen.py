import pandas as pd
import random
from faker import Faker

# Initialize Faker instance
fake = Faker()

# Define constants
NUM_TRANSACTIONS = 1000
NUM_CLIENTS = 10
MAX_TRANSACTION_AMOUNT = 2500
AGE_RANGE = (5, 60)
TRANSACTION_TYPES = ['Deposit', 'Withdrawal']
BRANCHES_ATM_LOCATIONS = ['NYC 6th Ave 44th St Branch', 'Brooklyn 4th Street Branch 2', 'Albany, NY, Main Street Branch 3', 'Staten Island ATM 1', 'Garden City, NY Main  Branch', 'Bay Shore, NY ATM 2']
DAYS = pd.date_range(start='2025-01-01', periods=5).tolist()

# Generate client details
clients = []
for i in range(NUM_CLIENTS):
    clients.append({
        'CustomerID': i + 1001,
        'CustomerName': fake.name(),
        'CustomerEthnicity': fake.random_element(elements=('Asian', 'Black', 'Hispanic', 'White', 'Other')),
        'CustomerAge': random.randint(*AGE_RANGE)
    })

# Generate transactions
transactions = []
for i in range(NUM_TRANSACTIONS):
    client = random.choice(clients)
    transaction_date = random.choice(DAYS)
    transaction = {
        'TransactionDate': transaction_date,
        'TransactionID': fake.uuid4(),
        'CustomerID': client['CustomerID'],
        'CustomerName': client['CustomerName'],
        'CustomerEthnicity': client['CustomerEthnicity'],
        'CustomerAge': client['CustomerAge'],
        'TransactionAmount': round(random.uniform(1, MAX_TRANSACTION_AMOUNT), 2),
        'transactionType': random.choice(TRANSACTION_TYPES),
        'TransactionBranch_ATMLocation': random.choice(BRANCHES_ATM_LOCATIONS)    }
    transactions.append(transaction)

# Create DataFrame and sort by CustomerID and TransactionDate
df = pd.DataFrame(transactions)
df.sort_values(by=['CustomerID', 'TransactionDate'], inplace=True)

# Save to CSV
output_file = 'transactions.csv'
df.to_csv(output_file, index=False)

print(f"Data has been generated and saved to {output_file}")