# Payment System

This document explains the Monero-based payment system used in the Compunir decentralized GPU training network. The payment system enables secure, private transactions between GPU providers and users requesting computational resources.

## Overview

Compunir uses Monero (XMR) as its native cryptocurrency for several key reasons:

1. **Privacy**: Monero provides enhanced privacy features for all participants
2. **Low Fees**: Reasonable transaction fees even for micro-payments
3. **Decentralization**: Fully decentralized currency aligns with network principles
4. **Security**: Strong cryptographic foundations

The payment system enables two primary types of transactions:
- **Job Payments**: Users pay for GPU computation time
- **Verification Rewards**: Nodes earn additional rewards for verifying computations

## Setting Up Your Wallet

### Recommended Wallets

You can use any of these Monero wallets:

1. **Feather Wallet** (Recommended for beginners)
   - Easy to use desktop interface
   - Built-in node connection
   - Available at [featherwallet.org](https://featherwallet.org)

2. **Official Monero GUI Wallet**
   - Full-featured desktop wallet
   - Available at [getmonero.org](https://getmonero.org)

3. **Monero CLI Wallet**
   - Command-line interface for advanced users
   - Required for RPC functionality
   - Included with Monero software package

### Wallet Configuration

#### Basic Wallet Setup

1. Download and install your chosen wallet
2. Create a new wallet (or restore from seed if you have an existing wallet)
3. Securely store your recovery seed phrase (25 words)
4. Copy your primary address - this will be used in your configuration

#### Setting Up Wallet RPC (Required for Automated Payments)

For automatic payments processing, you'll need to run the Monero wallet RPC:

```bash
# Create a dedicated wallet for the node
monero-wallet-cli --generate-new-wallet node_wallet

# Start the wallet RPC service (use your own username/password)
monero-wallet-rpc --wallet-file node_wallet --rpc-bind-ip 127.0.0.1 --rpc-bind-port 18082 --rpc-login username:password --daemon-address node.moneroworld.com:18089
```

Update your `local_config.yml` with these settings:

```yaml
wallet:
  address: "your_monero_address_here"  # Your wallet address
  rpc_url: "http://127.0.0.1:18082"    # RPC service URL
  rpc_username: "username"             # RPC username
  rpc_password: "password"             # RPC password
  min_payment: 0.01                    # Minimum XMR for payout
  base_rate: 0.001                     # Base XMR per GPU hour
```

## Earning XMR

As a GPU provider, you earn XMR in two ways:

### 1. Computation Rewards

You receive payment for GPU resources provided for distributed training jobs. Payment is calculated based on:

- **Base Rate**: Set in configuration (default: 0.001 XMR per GPU hour)
- **GPU Utilization**: Higher utilization means higher rewards
- **Job Duration**: Longer jobs mean more earnings
- **Job Difficulty**: Complex jobs may have higher rates

The payment formula is:
```
Payment = BaseRate × Duration × UtilizationFactor × QualityFactor
```

Where:
- `Duration` is measured in hours
- `UtilizationFactor` varies from 0.5 to 1.2 based on GPU utilization
- `QualityFactor` varies from 0.8 to 1.5 based on computation quality

### 2. Verification Rewards

You can earn additional XMR by verifying other nodes' computations:

- **Verification Base Rate**: Typically 10-20% of computation rate
- **Verification Quality**: Higher quality verifications earn more
- **Verification Volume**: Limited by reputation and stake

Verification payments are sent immediately upon successful verification.

## Payment Flow

### For GPU Providers

1. **Job Acceptance**: Your node accepts a job based on your configuration preferences
2. **Escrow Creation**: The payment is held in escrow on the network
3. **Execution**: Your node executes the machine learning job
4. **Verification**: Results are verified by other nodes
5. **Payment Release**: Upon verification, payment is released to your wallet
6. **Confirmation**: Payment is confirmed on the Monero blockchain

### For GPU Users (Paying for Computation)

1. **Job Submission**: Submit a job with payment details
2. **Escrow Deposit**: Send XMR to the provided escrow address
3. **Job Distribution**: Job is distributed to suitable nodes
4. **Verification**: Results are verified by multiple nodes
5. **Completion**: Verified results are returned to you

## Payment Security

The payment system includes several security mechanisms:

### Escrow System

All payments use an escrow system to ensure fairness:

1. Funds are locked in escrow before computation begins
2. Multiple verification nodes validate results
3. Payment is released only after successful verification
4. Automatic refunds if verification fails

### Sybil Attack Protection

To prevent payment-related attacks, the system implements:

1. **Minimum Stake**: Nodes must stake a minimum amount of XMR
2. **Reputation System**: Payment rates are affected by reputation
3. **Verification Cross-Checking**: Multiple independent verifiers
4. **Geographic Distribution**: Ensures verifiers are geographically diverse

## Payment Rates and Economics

### Default Payment Rates

| Job Type | Base Rate (XMR/hour) | Quality Multiplier | Typical Earnings |
|----------|----------------------|-------------------|------------------|
| Basic Training | 0.001 | 1.0 | 0.024 XMR/day |
| Fine-tuning | 0.0012 | 1.1 | 0.032 XMR/day |
| Large Model Training | 0.0015 | 1.2 | 0.043 XMR/day |
| Verification Work | 0.0002 | 1.0 | 0.005 XMR/day |

These rates are dynamic and adjust based on network conditions.

### Optimizing Earnings

To maximize your earnings:

1. **Maintain High Uptime**: More available time means more jobs
2. **Optimize GPU Performance**: Higher utilization earns more
3. **Build Reputation**: Higher reputation means better job assignments
4. **Perform Verifications**: Extra income from verification work
5. **Increase Stake**: Higher stake can increase earnings potential

## Monitoring Earnings

### Dashboard Monitoring

The dashboard provides real-time earnings information:

1. **Earnings Overview**: Total and pending earnings
2. **Earnings History**: Historical earnings charts
3. **Job-specific Earnings**: Earnings by job

### Command Line Monitoring

Check earnings via the command-line tool:

```bash
# Get total earnings
python -m decentralized_gpu.tools.earnings total

# Get earnings breakdown
python -m decentralized_gpu.tools.earnings breakdown

# Export earnings history to CSV
python -m decentralized_gpu.tools.earnings export --format csv
```

## Payment Troubleshooting

### Common Issues

#### Missing Payments

If you believe you're missing payments:

1. Check transaction status: `python -m decentralized_gpu.tools.verify_payment [job_id]`
2. Verify your wallet address in configuration
3. Check wallet synchronization status
4. Ensure your node completed the job successfully
5. Check that verification passed

#### Wallet Synchronization

If your wallet isn't showing the correct balance:

1. Ensure your wallet is fully synchronized
2. Check network connectivity
3. Verify with a block explorer that payments were sent
4. Restart the wallet RPC service if necessary

#### Payment Too Small

If payments seem too small:

1. Check your GPU utilization
2. Verify your base rate configuration
3. Look at job quality factors
4. Check market rates and adjust settings

### Support and Help

For payment-related issues:

1. Check logs in `logs/payment.log`
2. Run diagnostic tool: `python -m decentralized_gpu.tools.payment_diagnostic`
3. Visit our community forum for help
4. Open an issue on GitHub for persistent problems

## Advanced Payment Features

### Multi-wallet Support

For advanced users, multiple wallet support enables:

1. Separation of earnings and operational funds
2. Different wallets for different job types
3. Wallet rotation for enhanced privacy

Configure in `local_config.yml`:

```yaml
wallet:
  primary_address: "your_main_wallet_address"
  operational_address: "your_operational_wallet_address"
  verification_address: "your_verification_wallet_address"
  enable_wallet_rotation: true
```

### Payment Scheduling

Control when payments are processed:

```yaml
payment:
  auto_payout: true           # Automatic payouts
  min_payout_amount: 0.01     # Minimum amount for payout
  payout_interval: 86400      # Payout interval in seconds (24h)
  max_payout_per_day: 0.5     # Maximum daily payout
```

### Payment Reports

Generate detailed payment reports:

```bash
# Generate daily report
python -m decentralized_gpu.tools.payment_report daily

# Generate monthly report
python -m decentralized_gpu.tools.payment_report monthly --month 3 --year 2025

# Export for tax purposes
python -m decentralized_gpu.tools.payment_report tax --year 2025
```

## Regulatory Considerations

### Taxation

Earnings from the network may be subject to taxation:

1. In most jurisdictions, earnings are considered income
2. Keep detailed records of all transactions
3. Use the payment reporting tools for tax documentation
4. Consult a tax professional for advice specific to your location

### Compliance

As a decentralized system using privacy-focused cryptocurrency:

1. Ensure compliance with local regulations
2. Some jurisdictions have specific requirements for cryptocurrency activities
3. Not intended for circumventing legal obligations

## Future Payment System Enhancements

Planned future enhancements:

1. **Multi-currency Support**: Additional cryptocurrency options
2. **Payment Channels**: Lightning-like micro-payment channels for efficiency
3. **Enhanced Reputation System**: More nuanced reputation effects on payment
4. **Dynamic Rate Adjustment**: Automatic rate adjustment based on market conditions
5. **Smart Contract Integration**: Advanced payment conditions

## Reference

### Monero Resources

- [Official Monero Website](https://getmonero.org)
- [Feather Wallet](https://featherwallet.org)
- [Monero RPC Documentation](https://www.getmonero.org/resources/developer-guides/wallet-rpc.html)

### Payment-Related Commands

```bash
# Check wallet balance
python -m decentralized_gpu.tools.wallet balance

# List recent transactions
python -m decentralized_gpu.tools.wallet transactions

# View pending payments
python -m decentralized_gpu.tools.wallet pending

# Verify specific payment
python -m decentralized_gpu.tools.wallet verify [txid]
```

### Configuration Reference

For detailed payment configuration options, see the [Configuration Guide](./CONFIGURATION.md).