#!/usr/bin/env python3
"""
Blockchain Transaction Fee Evaluation
=====================================

This script evaluates the gas costs and transaction fees for the
BiometricAuth smart contract operations.

Operations evaluated:
1. User Registration (storing helper data on-chain)
2. Challenge Request
3. Authentication (signature verification)
4. Helper Data Retrieval (view, free)

Usage:
    python evaluate_blockchain.py                    # Full evaluation
    python evaluate_blockchain.py --eth-price 3500  # Custom ETH price
    python evaluate_blockchain.py --gas-price 30    # Custom gas price (gwei)
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class GasEstimate:
    """Gas cost estimate for a contract operation."""
    operation: str
    base_gas: int           # Fixed gas cost
    per_byte_gas: int       # Additional gas per byte of data
    typical_bytes: int      # Typical data size in bytes
    total_gas: int          # Total estimated gas
    description: str


@dataclass 
class CostEstimate:
    """Cost estimate in different currencies."""
    gas: int
    eth: float
    usd: float


class BlockchainGasEvaluator:
    """
    Evaluates gas costs for BiometricAuth smart contract.
    
    Gas cost breakdown:
    - Base transaction: 21,000 gas
    - Storage (SSTORE new): 20,000 gas per 32-byte slot
    - Storage (SSTORE update): 5,000 gas per slot
    - Memory operations: 3 gas per byte
    - Calldata: 16 gas per non-zero byte, 4 gas per zero byte
    - Keccak256: 30 + 6 per word
    - ECDSA recovery: ~3,000 gas
    - Event emission: 375 + 8 per byte (indexed: 375 per topic)
    """
    
    # Gas constants (based on EVM specifications)
    GAS_CONSTANTS = {
        'tx_base': 21000,           # Base transaction cost
        'sstore_new': 20000,        # New storage slot
        'sstore_update': 5000,      # Update existing slot
        'sload': 2100,              # Load from storage
        'memory_per_byte': 3,       # Memory expansion
        'calldata_nonzero': 16,     # Non-zero byte in calldata
        'calldata_zero': 4,         # Zero byte in calldata
        'keccak_base': 30,          # Hash base cost
        'keccak_per_word': 6,       # Hash per 32-byte word
        'ecrecover': 3000,          # ECDSA recovery
        'event_base': 375,          # Log event base
        'event_topic': 375,         # Per indexed topic
        'event_per_byte': 8,        # Per byte of event data
        'call_stipend': 2300,       # Call gas stipend
    }
    
    def __init__(
        self, 
        gas_price_gwei: float = 20.0,
        eth_price_usd: float = 3500.0
    ):
        """
        Initialize evaluator.
        
        Args:
            gas_price_gwei: Gas price in Gwei (1 Gwei = 10^-9 ETH)
            eth_price_usd: ETH price in USD
        """
        self.gas_price_gwei = gas_price_gwei
        self.eth_price_usd = eth_price_usd
    
    def estimate_registration_gas(self, helper_data_bytes: int = 100) -> GasEstimate:
        """
        Estimate gas for user registration.
        
        Storage layout for UserRecord:
        - helperData: dynamic bytes (1 slot for length + ceil(bytes/32) slots for data)
        - publicKeyHash: bytes32 (1 slot)
        - registrationTime: uint256 (1 slot)
        - isActive: bool (packed)
        - authCount: uint256 (1 slot)
        
        Args:
            helper_data_bytes: Size of helper data in bytes
        """
        g = self.GAS_CONSTANTS
        
        # Base transaction
        gas = g['tx_base']
        
        # Calldata (helper_data + publicKeyHash)
        # Assume ~10% zero bytes in helper data
        nonzero_bytes = int(helper_data_bytes * 0.9) + 32  # + publicKeyHash
        zero_bytes = int(helper_data_bytes * 0.1) + 4      # + function selector zeros
        gas += nonzero_bytes * g['calldata_nonzero']
        gas += zero_bytes * g['calldata_zero']
        
        # Storage: helperData
        # - Length slot: 1 new slot
        # - Data slots: ceil(helper_data_bytes / 32)
        helper_data_slots = (helper_data_bytes + 31) // 32 + 1
        gas += helper_data_slots * g['sstore_new']
        
        # Storage: publicKeyHash (1 slot)
        gas += g['sstore_new']
        
        # Storage: registrationTime (1 slot)
        gas += g['sstore_new']
        
        # Storage: isActive + authCount (can be packed, 1-2 slots)
        gas += g['sstore_new']
        
        # Storage: totalUsers increment (update existing)
        gas += g['sstore_update']
        
        # Event emission: UserRegistered
        # - 2 indexed topics (user, publicKeyHash)
        # - 1 data field (timestamp)
        gas += g['event_base'] + 2 * g['event_topic'] + 32 * g['event_per_byte']
        
        # Function overhead (memory, jumps, etc.)
        gas += 5000
        
        return GasEstimate(
            operation="register",
            base_gas=g['tx_base'] + 5000,
            per_byte_gas=g['calldata_nonzero'] + (g['sstore_new'] // 32),
            typical_bytes=helper_data_bytes,
            total_gas=gas,
            description=f"Register user with {helper_data_bytes} bytes helper data"
        )
    
    def estimate_request_challenge_gas(self) -> GasEstimate:
        """
        Estimate gas for requesting an authentication challenge.
        
        Operations:
        - Load user record (check isActive)
        - Compute keccak256 for nonce
        - Store challenge (nonce, expiryTime, used)
        - Emit event
        """
        g = self.GAS_CONSTANTS
        
        # Base transaction
        gas = g['tx_base']
        
        # Minimal calldata (just function selector)
        gas += 4 * g['calldata_nonzero']
        
        # Load user record to check isActive
        gas += g['sload']
        
        # Compute nonce (keccak256 of multiple values)
        gas += g['keccak_base'] + 4 * g['keccak_per_word']  # ~4 words input
        
        # Store challenge (3 fields, can fit in 2 slots)
        gas += 2 * g['sstore_new']  # First challenge write
        
        # Event: ChallengeIssued
        gas += g['event_base'] + 2 * g['event_topic'] + 32 * g['event_per_byte']
        
        # Function overhead
        gas += 3000
        
        return GasEstimate(
            operation="requestChallenge",
            base_gas=gas,
            per_byte_gas=0,
            typical_bytes=0,
            total_gas=gas,
            description="Request authentication challenge"
        )
    
    def estimate_authenticate_gas(self) -> GasEstimate:
        """
        Estimate gas for authentication (signature verification).
        
        Operations:
        - Load user record
        - Load challenge
        - Multiple condition checks
        - Compute message hash (keccak256)
        - ECDSA recover
        - Verify signature
        - Update challenge.used
        - Update usedNonces mapping
        - Update user.authCount
        - Update totalAuths
        - Emit event
        """
        g = self.GAS_CONSTANTS
        
        # Base transaction
        gas = g['tx_base']
        
        # Calldata (signature ~65 bytes)
        gas += 65 * g['calldata_nonzero'] + 4 * g['calldata_nonzero']  # sig + selector
        
        # Load user record (multiple slots)
        gas += 3 * g['sload']
        
        # Load challenge
        gas += 2 * g['sload']
        
        # Compute message hash
        gas += g['keccak_base'] + 3 * g['keccak_per_word']
        
        # ECDSA recovery (precompile)
        gas += g['ecrecover']
        
        # Verify signature (another keccak for comparison)
        gas += g['keccak_base'] + g['keccak_per_word']
        
        # Update challenge.used
        gas += g['sstore_update']
        
        # Update usedNonces mapping
        gas += g['sstore_new']  # New entry in mapping
        
        # Update user.authCount
        gas += g['sstore_update']
        
        # Update totalAuths
        gas += g['sstore_update']
        
        # Event: AuthenticationSuccess
        gas += g['event_base'] + g['event_topic'] + 64 * g['event_per_byte']
        
        # Function overhead
        gas += 5000
        
        return GasEstimate(
            operation="authenticate",
            base_gas=gas,
            per_byte_gas=0,
            typical_bytes=65,  # Signature size
            total_gas=gas,
            description="Authenticate with signature verification"
        )
    
    def estimate_get_helper_data_gas(self) -> GasEstimate:
        """
        Estimate gas for getHelperData (view function - FREE for users).
        
        Note: View functions don't cost gas when called externally,
        but we estimate execution cost for reference.
        """
        g = self.GAS_CONSTANTS
        
        # View functions are free for external calls
        # But compute execution cost for reference
        gas = g['sload'] * 3  # Load user record + helper data
        gas += 2000  # Memory operations
        
        return GasEstimate(
            operation="getHelperData",
            base_gas=0,  # Free for view calls
            per_byte_gas=0,
            typical_bytes=0,
            total_gas=0,  # FREE
            description="Retrieve helper data (view function - FREE)"
        )
    
    def gas_to_cost(self, gas: int) -> CostEstimate:
        """Convert gas to ETH and USD costs."""
        eth = gas * self.gas_price_gwei * 1e-9
        usd = eth * self.eth_price_usd
        return CostEstimate(gas=gas, eth=eth, usd=usd)
    
    def evaluate_all(self, helper_data_sizes: List[int] = None) -> Dict:
        """
        Evaluate gas costs for all operations.
        
        Args:
            helper_data_sizes: List of helper data sizes to test
            
        Returns:
            Dictionary with all evaluation results
        """
        if helper_data_sizes is None:
            helper_data_sizes = [100, 150, 200, 300]  # Typical sizes
        
        results = {
            'parameters': {
                'gas_price_gwei': self.gas_price_gwei,
                'eth_price_usd': self.eth_price_usd,
            },
            'operations': {},
            'registration_by_size': {},
            'total_flow': {},
        }
        
        # Fixed operations
        challenge = self.estimate_request_challenge_gas()
        auth = self.estimate_authenticate_gas()
        view = self.estimate_get_helper_data_gas()
        
        results['operations']['requestChallenge'] = {
            'gas': challenge.total_gas,
            **self.gas_to_cost(challenge.total_gas).__dict__,
            'description': challenge.description
        }
        
        results['operations']['authenticate'] = {
            'gas': auth.total_gas,
            **self.gas_to_cost(auth.total_gas).__dict__,
            'description': auth.description
        }
        
        results['operations']['getHelperData'] = {
            'gas': view.total_gas,
            **self.gas_to_cost(view.total_gas).__dict__,
            'description': view.description
        }
        
        # Registration by helper data size
        for size in helper_data_sizes:
            reg = self.estimate_registration_gas(size)
            cost = self.gas_to_cost(reg.total_gas)
            results['registration_by_size'][size] = {
                'gas': reg.total_gas,
                **cost.__dict__,
                'description': reg.description
            }
        
        # Typical registration (100 bytes)
        reg_100 = self.estimate_registration_gas(100)
        results['operations']['register'] = {
            'gas': reg_100.total_gas,
            **self.gas_to_cost(reg_100.total_gas).__dict__,
            'description': f"Register user (100 bytes helper data)"
        }
        
        # Complete authentication flow
        # One-time: registration
        # Per-auth: requestChallenge + authenticate
        reg_gas = reg_100.total_gas
        auth_flow_gas = challenge.total_gas + auth.total_gas
        
        results['total_flow'] = {
            'enrollment': {
                'gas': reg_gas,
                **self.gas_to_cost(reg_gas).__dict__,
                'description': 'One-time registration'
            },
            'authentication': {
                'gas': auth_flow_gas,
                **self.gas_to_cost(auth_flow_gas).__dict__,
                'description': 'Per-authentication (challenge + verify)'
            }
        }
        
        return results


def print_evaluation_report(results: Dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 75)
    print("BLOCKCHAIN TRANSACTION FEE EVALUATION")
    print("BiometricAuth Smart Contract")
    print("=" * 75)
    
    params = results['parameters']
    print(f"\n[PARAMETERS]")
    print(f"  Gas Price: {params['gas_price_gwei']} Gwei")
    print(f"  ETH Price: ${params['eth_price_usd']:,.2f} USD")
    
    # Operations summary
    print(f"\n" + "-" * 75)
    print("OPERATION GAS COSTS")
    print("-" * 75)
    print(f"{'Operation':<25} {'Gas':>12} {'ETH':>15} {'USD':>12}")
    print("-" * 75)
    
    for op_name, op_data in results['operations'].items():
        gas = op_data['gas']
        eth = op_data['eth']
        usd = op_data['usd']
        if gas > 0:
            print(f"{op_name:<25} {gas:>12,} {eth:>15.8f} ${usd:>10.4f}")
        else:
            print(f"{op_name:<25} {'FREE':>12} {'FREE':>15} {'FREE':>12}")
    
    print("-" * 75)
    
    # Registration by size
    print(f"\n" + "-" * 75)
    print("REGISTRATION COST BY HELPER DATA SIZE")
    print("-" * 75)
    print(f"{'Helper Data (bytes)':<25} {'Gas':>12} {'ETH':>15} {'USD':>12}")
    print("-" * 75)
    
    for size, data in results['registration_by_size'].items():
        print(f"{size:<25} {data['gas']:>12,} {data['eth']:>15.8f} ${data['usd']:>10.4f}")
    
    print("-" * 75)
    
    # Total flow costs
    print(f"\n" + "-" * 75)
    print("COMPLETE AUTHENTICATION FLOW COSTS")
    print("-" * 75)
    
    enroll = results['total_flow']['enrollment']
    auth = results['total_flow']['authentication']
    
    print(f"\n[ENROLLMENT] (One-time cost)")
    print(f"  Gas:  {enroll['gas']:,}")
    print(f"  ETH:  {enroll['eth']:.8f}")
    print(f"  USD:  ${enroll['usd']:.4f}")
    
    print(f"\n[AUTHENTICATION] (Per-login cost)")
    print(f"  Gas:  {auth['gas']:,}")
    print(f"  ETH:  {auth['eth']:.8f}")
    print(f"  USD:  ${auth['usd']:.4f}")
    
    # Cost projections
    print(f"\n" + "-" * 75)
    print("COST PROJECTIONS")
    print("-" * 75)
    
    auth_per_day = [1, 5, 10, 50]
    print(f"\n{'Logins/Day':<15} {'Daily Cost':>15} {'Monthly Cost':>15} {'Yearly Cost':>15}")
    print("-" * 75)
    
    for logins in auth_per_day:
        daily = logins * auth['usd']
        monthly = daily * 30
        yearly = daily * 365
        print(f"{logins:<15} ${daily:>14.2f} ${monthly:>14.2f} ${yearly:>14.2f}")
    
    print("-" * 75)
    
    # Comparison with other systems
    print(f"\n" + "-" * 75)
    print("COMPARISON WITH OTHER AUTHENTICATION SYSTEMS")
    print("-" * 75)
    
    comparisons = [
        ("Traditional Password DB", "Free", "$0.001/query (cloud DB)"),
        ("OAuth/OIDC Provider", "Free", "$0.01-0.05/auth (Auth0)"),
        ("SMS OTP", "Free", "$0.01-0.05/SMS"),
        ("Hardware Token (YubiKey)", "$50 upfront", "Free per auth"),
        ("Blockchain (Our System)", f"${enroll['usd']:.2f} enroll", f"${auth['usd']:.4f}/auth"),
    ]
    
    print(f"\n{'System':<30} {'Setup Cost':<20} {'Per-Auth Cost':<25}")
    print("-" * 75)
    for system, setup, per_auth in comparisons:
        print(f"{system:<30} {setup:<20} {per_auth:<25}")
    
    print("-" * 75)
    
    # Network comparison
    print(f"\n" + "-" * 75)
    print("COST COMPARISON ACROSS NETWORKS")
    print("-" * 75)
    
    networks = [
        ("Ethereum Mainnet", 20, 3500),
        ("Ethereum (High Gas)", 100, 3500),
        ("Polygon", 50, 0.80),
        ("Arbitrum", 0.1, 3500),
        ("Optimism", 0.01, 3500),
        ("BSC", 5, 300),
        ("Local/Testnet", 0, 0),
    ]
    
    reg_gas = results['operations']['register']['gas']
    auth_gas = results['total_flow']['authentication']['gas']
    
    print(f"\n{'Network':<25} {'Gas Price':>12} {'ETH Price':>12} {'Register':>12} {'Auth':>12}")
    print("-" * 75)
    
    for network, gas_price, eth_price in networks:
        reg_cost = reg_gas * gas_price * 1e-9 * eth_price
        auth_cost = auth_gas * gas_price * 1e-9 * eth_price
        
        if eth_price == 0:
            print(f"{network:<25} {gas_price:>10} Gwei {'$0':>12} {'FREE':>12} {'FREE':>12}")
        else:
            print(f"{network:<25} {gas_price:>10} Gwei ${eth_price:>10,.0f} ${reg_cost:>10.4f} ${auth_cost:>10.4f}")
    
    print("-" * 75)
    
    # Summary
    print(f"\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"""
  • Registration (one-time):     ~{enroll['gas']:,} gas = ${enroll['usd']:.4f}
  • Authentication (per-login):  ~{auth['gas']:,} gas = ${auth['usd']:.4f}
  • Helper data storage:         ~100-200 bytes on-chain
  • View functions:              FREE (no gas cost)
  
  RECOMMENDATIONS:
  • For production: Consider Layer 2 (Polygon, Arbitrum) for 10-100x lower costs
  • For development: Use local testnet (free)
  • Break-even vs SMS OTP: ~{0.03 / auth['usd']:.0f} authentications
    """)
    
    print("=" * 75)


def print_benchmark_table(results: Dict):
    """Print a clean benchmark comparison table."""
    print("\n" + "=" * 75)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 75)
    
    reg = results['operations']['register']
    challenge = results['operations']['requestChallenge']
    auth = results['operations']['authenticate']
    view = results['operations']['getHelperData']
    total_auth = results['total_flow']['authentication']
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    BLOCKCHAIN GAS COST BENCHMARK                         │
├─────────────────────────┬──────────────┬───────────────┬────────────────┤
│ Operation               │ Gas          │ ETH           │ USD            │
├─────────────────────────┼──────────────┼───────────────┼────────────────┤
│ register()              │ {reg['gas']:>10,} │ {reg['eth']:>13.8f} │ ${reg['usd']:>12.4f} │
│ requestChallenge()      │ {challenge['gas']:>10,} │ {challenge['eth']:>13.8f} │ ${challenge['usd']:>12.4f} │
│ authenticate()          │ {auth['gas']:>10,} │ {auth['eth']:>13.8f} │ ${auth['usd']:>12.4f} │
│ getHelperData() [view]  │ {'FREE':>10} │ {'FREE':>13} │ {'FREE':>13} │
├─────────────────────────┼──────────────┼───────────────┼────────────────┤
│ ENROLLMENT (one-time)   │ {reg['gas']:>10,} │ {reg['eth']:>13.8f} │ ${reg['usd']:>12.4f} │
│ AUTH FLOW (per-login)   │ {total_auth['gas']:>10,} │ {total_auth['eth']:>13.8f} │ ${total_auth['usd']:>12.4f} │
└─────────────────────────┴──────────────┴───────────────┴────────────────┘
    """)
    
    print(f"Parameters: Gas Price = {results['parameters']['gas_price_gwei']} Gwei, ETH = ${results['parameters']['eth_price_usd']:,.0f}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate blockchain transaction fees for BiometricAuth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_blockchain.py                     # Default evaluation
  python evaluate_blockchain.py --eth-price 4000   # Custom ETH price
  python evaluate_blockchain.py --gas-price 50     # High gas scenario
  python evaluate_blockchain.py --l2               # Layer 2 costs
        """
    )
    
    parser.add_argument(
        '--gas-price', 
        type=float, 
        default=20.0,
        help='Gas price in Gwei (default: 20)'
    )
    
    parser.add_argument(
        '--eth-price', 
        type=float, 
        default=3500.0,
        help='ETH price in USD (default: 3500)'
    )
    
    parser.add_argument(
        '--l2',
        action='store_true',
        help='Use Layer 2 typical costs (0.1 Gwei)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Adjust for L2 if specified
    gas_price = 0.1 if args.l2 else args.gas_price
    
    # Create evaluator
    evaluator = BlockchainGasEvaluator(
        gas_price_gwei=gas_price,
        eth_price_usd=args.eth_price
    )
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_evaluation_report(results)
        print_benchmark_table(results)


if __name__ == "__main__":
    main()
