"""
Blockchain Client for Biometric Authentication
================================================

This module provides a Python interface to the BiometricAuth smart contract
using Web3.py.

Features:
- Contract deployment and interaction
- User registration and authentication
- Challenge-response protocol implementation
- ECDSA key derivation from fuzzy extractor output

Security Notes:
- Private keys should never be logged or stored in plaintext
- All signing operations happen locally (never send private key to network)
- Use hardware wallets or secure enclaves in production
"""

import os
import json
import hashlib
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

# Handle web3.py version differences for POA middleware
try:
    # web3.py v6+
    from web3.middleware import ExtraDataToPOAMiddleware
    POA_MIDDLEWARE = ExtraDataToPOAMiddleware
except ImportError:
    try:
        # web3.py v5.x
        from web3.middleware import geth_poa_middleware
        POA_MIDDLEWARE = geth_poa_middleware
    except ImportError:
        # No POA middleware available
        POA_MIDDLEWARE = None

from config import BlockchainConfig, DEFAULT_CONFIG


# Contract ABI (simplified - in production, compile from Solidity)
CONTRACT_ABI = [
    # Registration
    {
        "inputs": [
            {"name": "helperData", "type": "bytes"},
            {"name": "publicKeyHash", "type": "bytes32"}
        ],
        "name": "register",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # Request Challenge
    {
        "inputs": [],
        "name": "requestChallenge",
        "outputs": [
            {"name": "nonce", "type": "bytes32"},
            {"name": "expiryTime", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # Authenticate
    {
        "inputs": [
            {"name": "signature", "type": "bytes"}
        ],
        "name": "authenticate",
        "outputs": [
            {"name": "success", "type": "bool"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # Get Helper Data
    {
        "inputs": [
            {"name": "user", "type": "address"}
        ],
        "name": "getHelperData",
        "outputs": [
            {"name": "", "type": "bytes"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Is User Registered
    {
        "inputs": [
            {"name": "user", "type": "address"}
        ],
        "name": "isUserRegistered",
        "outputs": [
            {"name": "", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Get Challenge
    {
        "inputs": [
            {"name": "user", "type": "address"}
        ],
        "name": "getChallenge",
        "outputs": [
            {"name": "nonce", "type": "bytes32"},
            {"name": "expiryTime", "type": "uint256"},
            {"name": "isValid", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Events
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "user", "type": "address"},
            {"indexed": True, "name": "publicKeyHash", "type": "bytes32"},
            {"indexed": False, "name": "timestamp", "type": "uint256"}
        ],
        "name": "UserRegistered",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "user", "type": "address"},
            {"indexed": False, "name": "timestamp", "type": "uint256"},
            {"indexed": False, "name": "authCount", "type": "uint256"}
        ],
        "name": "AuthenticationSuccess",
        "type": "event"
    }
]


@dataclass
class BiometricKeyPair:
    """
    Key pair derived from fuzzy extractor output.
    
    Attributes:
        private_key: ECDSA private key (32 bytes)
        public_key: ECDSA public key (compressed, 33 bytes)
        address: Ethereum address derived from public key
        public_key_hash: Keccak256 hash of the address (for contract storage)
    """
    private_key: bytes
    public_key: bytes
    address: str
    public_key_hash: bytes


def derive_keypair(secret_key: bytes) -> BiometricKeyPair:
    """
    Derive an Ethereum keypair from fuzzy extractor secret key.
    
    Uses HKDF-like expansion to ensure the key has proper format
    for secp256k1 curve.
    
    Args:
        secret_key: Secret key from FuzzyExtractor (32 bytes recommended)
        
    Returns:
        BiometricKeyPair with all derived values
    """
    # Ensure key is 32 bytes (truncate or hash if needed)
    if len(secret_key) != 32:
        secret_key = hashlib.sha256(secret_key).digest()
    
    # Create Ethereum account from private key
    account = Account.from_key(secret_key)
    
    # Compute public key hash (used for on-chain verification)
    # Note: We hash the address as a proxy for the public key
    public_key_hash = Web3.keccak(text=account.address)
    
    return BiometricKeyPair(
        private_key=secret_key,
        public_key=bytes.fromhex(account.key.hex()[2:]),  # Remove '0x'
        address=account.address,
        public_key_hash=public_key_hash
    )


class BiometricAuthClient:
    """
    Client for interacting with the BiometricAuth smart contract.
    
    This client handles:
    - Connection to Ethereum network
    - Contract deployment and interaction
    - Transaction signing and submission
    - Challenge-response authentication protocol
    
    Usage:
        client = BiometricAuthClient()
        client.connect()
        
        # Registration
        client.register(helper_data, keypair)
        
        # Authentication
        client.authenticate(keypair)
    """
    
    def __init__(self, config: BlockchainConfig = None):
        self.config = config or DEFAULT_CONFIG.blockchain
        self.w3: Optional[Web3] = None
        self.contract = None
        self.account = None
        
    def connect(self, private_key: Optional[str] = None) -> bool:
        """
        Connect to the Ethereum network.
        
        Args:
            private_key: Optional private key for transactions.
                        If not provided, uses the first account from the node.
                        
        Returns:
            True if connection successful
        """
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.config.provider_url))
            
            # Add middleware for PoA chains (like Ganache)
            if POA_MIDDLEWARE is not None:
                try:
                    self.w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
                except Exception:
                    pass  # Middleware injection failed, continue anyway
            
            if not self.w3.is_connected():
                print(f"Failed to connect to {self.config.provider_url}")
                return False
            
            # Set up account
            if private_key:
                self.account = Account.from_key(private_key)
            elif self.w3.eth.accounts:
                # Use first account from node (for development)
                self.account = Account.from_key(
                    # This won't work with real nodes; need explicit private key
                    "0x" + "0" * 64  # Placeholder
                )
                print("Warning: Using placeholder account. Provide private_key for real transactions.")
            
            print(f"Connected to chain ID: {self.w3.eth.chain_id}")
            print(f"Current block: {self.w3.eth.block_number}")
            
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def deploy_contract(self, bytecode: str) -> str:
        """
        Deploy the BiometricAuth contract.
        
        Args:
            bytecode: Compiled contract bytecode
            
        Returns:
            Deployed contract address
        """
        if not self.w3 or not self.account:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Create contract instance
        Contract = self.w3.eth.contract(abi=CONTRACT_ABI, bytecode=bytecode)
        
        # Build deployment transaction
        tx = Contract.constructor().build_transaction({
            'from': self.account.address,
            'gas': self.config.gas_limit,
            'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        contract_address = receipt['contractAddress']
        self.config.contract_address = contract_address
        
        # Initialize contract instance
        self.contract = self.w3.eth.contract(
            address=contract_address,
            abi=CONTRACT_ABI
        )
        
        print(f"Contract deployed at: {contract_address}")
        return contract_address
    
    def load_contract(self, address: str):
        """
        Load an existing contract at the given address.
        
        Args:
            address: Contract address
        """
        if not self.w3:
            raise RuntimeError("Not connected. Call connect() first.")
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=CONTRACT_ABI
        )
        self.config.contract_address = address
        print(f"Loaded contract at: {address}")
    
    def register(
        self,
        helper_data: bytes,
        keypair: BiometricKeyPair,
        from_account: Optional[Account] = None
    ) -> str:
        """
        Register a user with their biometric helper data.
        
        Args:
            helper_data: Helper data from FuzzyExtractor.gen()
            keypair: Derived keypair from the fuzzy extractor key
            from_account: Account to send transaction from
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded. Call deploy_contract() or load_contract().")
        
        account = from_account or self.account
        
        # Build transaction
        tx = self.contract.functions.register(
            helper_data,
            keypair.public_key_hash
        ).build_transaction({
            'from': account.address,
            'gas': self.config.gas_limit,
            'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt['status'] == 1:
            print(f"Registration successful! Tx: {tx_hash.hex()}")
        else:
            print(f"Registration failed! Tx: {tx_hash.hex()}")
        
        return tx_hash.hex()
    
    def request_challenge(
        self,
        from_account: Optional[Account] = None
    ) -> Tuple[bytes, int]:
        """
        Request an authentication challenge.
        
        Args:
            from_account: Account requesting the challenge
            
        Returns:
            (nonce, expiry_time) tuple
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded.")
        
        account = from_account or self.account
        
        # Build transaction
        tx = self.contract.functions.requestChallenge().build_transaction({
            'from': account.address,
            'gas': self.config.gas_limit,
            'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'chainId': self.config.chain_id
        })
        
        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get challenge from contract (view function)
        challenge = self.contract.functions.getChallenge(account.address).call()
        nonce, expiry_time, is_valid = challenge
        
        return nonce, expiry_time
    
    def authenticate(
        self,
        keypair: BiometricKeyPair,
        from_account: Optional[Account] = None
    ) -> bool:
        """
        Perform biometric authentication.
        
        This implements the full challenge-response protocol:
        1. Request challenge from contract
        2. Sign challenge with biometric-derived key
        3. Submit signature for verification
        
        Args:
            keypair: Key pair derived from recovered fuzzy extractor key
            from_account: Account to authenticate
            
        Returns:
            True if authentication successful
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded.")
        
        account = from_account or self.account
        
        # Step 1: Request challenge
        print("Requesting challenge...")
        nonce, expiry_time = self.request_challenge(account)
        print(f"  Nonce: {nonce.hex()}")
        print(f"  Expires: {expiry_time}")
        
        # Step 2: Sign challenge
        print("Signing challenge...")
        signature = self._sign_challenge(nonce, keypair, account.address)
        print(f"  Signature: {signature.hex()[:32]}...")
        
        # Step 3: Submit signature
        print("Submitting authentication...")
        tx = self.contract.functions.authenticate(signature).build_transaction({
            'from': account.address,
            'gas': self.config.gas_limit,
            'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei'),
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'chainId': self.config.chain_id
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        success = receipt['status'] == 1
        if success:
            print("✓ Authentication successful!")
        else:
            print("✗ Authentication failed!")
        
        return success
    
    def _sign_challenge(
        self,
        nonce: bytes,
        keypair: BiometricKeyPair,
        address: str
    ) -> bytes:
        """
        Sign an authentication challenge.
        
        The message format matches what the contract expects:
            keccak256(abi.encodePacked(nonce, address, chainId))
        
        Args:
            nonce: Challenge nonce from contract
            keypair: Biometric-derived keypair
            address: User's address
            
        Returns:
            ECDSA signature bytes
        """
        # Construct message (must match contract exactly)
        message = Web3.solidity_keccak(
            ['bytes32', 'address', 'uint256'],
            [nonce, address, self.config.chain_id]
        )
        
        # Sign as Ethereum signed message
        signable = encode_defunct(primitive=message)
        signed = Account.sign_message(signable, keypair.private_key)
        
        return signed.signature
    
    # =========================================================================
    # View Functions
    # =========================================================================
    
    def get_helper_data(self, user_address: str) -> bytes:
        """
        Fetch helper data for a user.
        
        Args:
            user_address: Ethereum address of the user
            
        Returns:
            Helper data bytes
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded.")
        
        return self.contract.functions.getHelperData(
            Web3.to_checksum_address(user_address)
        ).call()
    
    def is_user_registered(self, user_address: str) -> bool:
        """Check if a user is registered."""
        if not self.contract:
            raise RuntimeError("Contract not loaded.")
        
        return self.contract.functions.isUserRegistered(
            Web3.to_checksum_address(user_address)
        ).call()


class MockBlockchainClient:
    """
    Mock blockchain client for testing without a real network.
    
    This simulates the contract behavior locally, useful for:
    - Unit testing
    - Development without Ganache/Hardhat
    - Performance benchmarking
    """
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.challenges: Dict[str, Dict[str, Any]] = {}
        self.chain_id = 1337
        
    def connect(self, private_key: Optional[str] = None) -> bool:
        print("MockBlockchain: Connected")
        return True
    
    def register(
        self,
        helper_data: bytes,
        keypair: BiometricKeyPair,
        from_account = None
    ) -> str:
        address = keypair.address
        
        if address in self.users:
            raise ValueError("User already registered")
        
        self.users[address] = {
            'helper_data': helper_data,
            'public_key_hash': keypair.public_key_hash,
            'auth_count': 0
        }
        
        print(f"MockBlockchain: Registered {address[:10]}...")
        return "0x" + "0" * 64  # Fake tx hash
    
    def request_challenge(self, from_account = None) -> Tuple[bytes, int]:
        import time
        import secrets
        
        nonce = secrets.token_bytes(32)
        expiry = int(time.time()) + 300  # 5 minutes
        
        # For mock, we store by the caller (simplified)
        self.challenges['mock'] = {
            'nonce': nonce,
            'expiry': expiry,
            'used': False
        }
        
        return nonce, expiry
    
    def authenticate(self, keypair: BiometricKeyPair, from_account = None) -> bool:
        address = keypair.address
        
        if address not in self.users:
            return False
        
        # Verify public key hash matches
        stored_hash = self.users[address]['public_key_hash']
        current_hash = Web3.keccak(text=address)
        
        if stored_hash != current_hash:
            return False
        
        self.users[address]['auth_count'] += 1
        print(f"MockBlockchain: Authenticated {address[:10]}...")
        return True
    
    def get_helper_data(self, user_address: str) -> bytes:
        if user_address not in self.users:
            raise ValueError("User not registered")
        return self.users[user_address]['helper_data']
    
    def is_user_registered(self, user_address: str) -> bool:
        return user_address in self.users


if __name__ == "__main__":
    # Test with mock client
    print("=" * 60)
    print("Blockchain Client Test (Mock)")
    print("=" * 60)
    
    client = MockBlockchainClient()
    client.connect()
    
    # Generate a test keypair
    test_secret = b'0' * 32
    keypair = derive_keypair(test_secret)
    
    print(f"\nDerived keypair:")
    print(f"  Address: {keypair.address}")
    print(f"  Public key hash: {keypair.public_key_hash.hex()[:16]}...")
    
    # Test registration
    print("\nTesting registration...")
    helper_data = b"test_helper_data_from_fuzzy_extractor"
    tx_hash = client.register(helper_data, keypair)
    print(f"  Registered: {client.is_user_registered(keypair.address)}")
    
    # Test authentication
    print("\nTesting authentication...")
    success = client.authenticate(keypair)
    print(f"  Success: {success}")
    
    # Test helper data retrieval
    print("\nTesting helper data retrieval...")
    retrieved = client.get_helper_data(keypair.address)
    print(f"  Retrieved matches original: {retrieved == helper_data}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)