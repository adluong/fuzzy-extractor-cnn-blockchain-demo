// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title BiometricAuth
 * @author CNN-FE-Blockchain Research
 * @notice Decentralized biometric authentication using fuzzy extractors
 * @dev Stores helper data on-chain; key derivation happens off-chain
 * 
 * Security Model:
 * - Helper data P is public but reveals no information about the key
 * - Authentication requires possession of the biometric AND the correct key
 * - Challenge-response prevents replay attacks
 * 
 * Privacy Guarantees:
 * - No raw biometric data is ever stored on-chain
 * - Helper data is computationally unlinkable to the biometric
 * - User addresses can be pseudonymous
 */

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

contract BiometricAuth is ReentrancyGuard, Ownable {
    using ECDSA for bytes32;
    
    // =========================================================================
    // Type Definitions
    // =========================================================================
    
    /**
     * @notice User registration data
     * @param helperData The fuzzy extractor helper data P
     * @param publicKeyHash Hash of the derived public key for signature verification
     * @param registrationTime Block timestamp of registration
     * @param isActive Whether the registration is currently active
     * @param authCount Number of successful authentications
     */
    struct UserRecord {
        bytes helperData;
        bytes32 publicKeyHash;
        uint256 registrationTime;
        bool isActive;
        uint256 authCount;
    }
    
    /**
     * @notice Authentication challenge
     * @param nonce Random challenge value
     * @param expiryTime Challenge expiration timestamp
     * @param used Whether the challenge has been used
     */
    struct Challenge {
        bytes32 nonce;
        uint256 expiryTime;
        bool used;
    }
    
    // =========================================================================
    // State Variables
    // =========================================================================
    
    /// @notice Mapping from user address to their registration record
    mapping(address => UserRecord) public users;
    
    /// @notice Mapping from user address to their active challenge
    mapping(address => Challenge) public challenges;
    
    /// @notice Mapping to track used nonces (prevents replay)
    mapping(bytes32 => bool) public usedNonces;
    
    /// @notice Challenge validity duration (5 minutes)
    uint256 public constant CHALLENGE_DURATION = 5 minutes;
    
    /// @notice Protocol version for future upgrades
    uint8 public constant PROTOCOL_VERSION = 1;
    
    /// @notice Total registered users
    uint256 public totalUsers;
    
    /// @notice Total successful authentications
    uint256 public totalAuths;
    
    // =========================================================================
    // Events
    // =========================================================================
    
    event UserRegistered(
        address indexed user, 
        bytes32 indexed publicKeyHash,
        uint256 timestamp
    );
    
    event UserUpdated(
        address indexed user,
        bytes32 indexed newPublicKeyHash,
        uint256 timestamp
    );
    
    event UserDeactivated(
        address indexed user,
        uint256 timestamp
    );
    
    event ChallengeIssued(
        address indexed user,
        bytes32 indexed challengeHash,
        uint256 expiryTime
    );
    
    event AuthenticationSuccess(
        address indexed user,
        uint256 timestamp,
        uint256 authCount
    );
    
    event AuthenticationFailed(
        address indexed user,
        string reason,
        uint256 timestamp
    );
    
    // =========================================================================
    // Errors
    // =========================================================================
    
    error UserAlreadyRegistered();
    error UserNotRegistered();
    error UserNotActive();
    error InvalidHelperData();
    error ChallengeExpired();
    error ChallengeNotFound();
    error ChallengeAlreadyUsed();
    error InvalidSignature();
    error NonceAlreadyUsed();
    
    // =========================================================================
    // Constructor
    // =========================================================================
    
    constructor() Ownable(msg.sender) {}
    
    // =========================================================================
    // Registration Functions
    // =========================================================================
    
    /**
     * @notice Register a new user with their biometric helper data
     * @param helperData The fuzzy extractor helper data P from Gen()
     * @param publicKeyHash Keccak256 hash of the derived public key
     * @dev Helper data is stored as-is; no on-chain processing
     * 
     * The registration flow:
     * 1. User captures biometric off-chain
     * 2. CNN extracts embedding, BioHasher binarizes
     * 3. FuzzyExtractor.Gen() produces (key, helperData)
     * 4. Derive keypair from key, compute hash of public key
     * 5. Call this function with helperData and publicKeyHash
     */
    function register(
        bytes calldata helperData,
        bytes32 publicKeyHash
    ) external {
        if (users[msg.sender].isActive) {
            revert UserAlreadyRegistered();
        }
        
        if (helperData.length == 0 || helperData.length > 1024) {
            revert InvalidHelperData();
        }
        
        users[msg.sender] = UserRecord({
            helperData: helperData,
            publicKeyHash: publicKeyHash,
            registrationTime: block.timestamp,
            isActive: true,
            authCount: 0
        });
        
        totalUsers++;
        
        emit UserRegistered(msg.sender, publicKeyHash, block.timestamp);
    }
    
    /**
     * @notice Update user's biometric registration (re-enrollment)
     * @param newHelperData New helper data from fresh biometric
     * @param newPublicKeyHash Hash of the new derived public key
     * @param signature Signature proving possession of current key
     * @dev Requires authentication with current key before update
     */
    function updateRegistration(
        bytes calldata newHelperData,
        bytes32 newPublicKeyHash,
        bytes calldata signature
    ) external nonReentrant {
        UserRecord storage user = users[msg.sender];
        
        if (!user.isActive) {
            revert UserNotRegistered();
        }
        
        if (newHelperData.length == 0 || newHelperData.length > 1024) {
            revert InvalidHelperData();
        }
        
        // Verify signature to prove current key possession
        bytes32 messageHash = keccak256(abi.encodePacked(
            "UPDATE:",
            msg.sender,
            newPublicKeyHash,
            block.chainid
        ));
        
        if (!_verifySignature(messageHash, signature, user.publicKeyHash)) {
            revert InvalidSignature();
        }
        
        // Update registration
        user.helperData = newHelperData;
        user.publicKeyHash = newPublicKeyHash;
        
        emit UserUpdated(msg.sender, newPublicKeyHash, block.timestamp);
    }
    
    /**
     * @notice Deactivate user registration
     * @dev Only the user themselves or contract owner can deactivate
     */
    function deactivate() external {
        UserRecord storage user = users[msg.sender];
        
        if (!user.isActive) {
            revert UserNotRegistered();
        }
        
        user.isActive = false;
        totalUsers--;
        
        emit UserDeactivated(msg.sender, block.timestamp);
    }
    
    // =========================================================================
    // Authentication Functions
    // =========================================================================
    
    /**
     * @notice Request a challenge for authentication
     * @return nonce The challenge nonce to sign
     * @return expiryTime When the challenge expires
     * 
     * Authentication flow:
     * 1. User calls requestChallenge()
     * 2. Contract returns random nonce
     * 3. Off-chain: User captures biometric, recovers key via Rep()
     * 4. User signs the challenge with derived key
     * 5. User calls authenticate() with signature
     */
    function requestChallenge() external returns (bytes32 nonce, uint256 expiryTime) {
        UserRecord storage user = users[msg.sender];
        
        if (!user.isActive) {
            revert UserNotRegistered();
        }
        
        // Generate random nonce using block data and user address
        // Note: For production, use Chainlink VRF or similar
        nonce = keccak256(abi.encodePacked(
            block.timestamp,
            block.prevrandao,
            msg.sender,
            user.authCount
        ));
        
        expiryTime = block.timestamp + CHALLENGE_DURATION;
        
        challenges[msg.sender] = Challenge({
            nonce: nonce,
            expiryTime: expiryTime,
            used: false
        });
        
        emit ChallengeIssued(msg.sender, keccak256(abi.encodePacked(nonce)), expiryTime);
        
        return (nonce, expiryTime);
    }
    
    /**
     * @notice Authenticate by providing a valid signature of the challenge
     * @param signature ECDSA signature of the challenge nonce
     * @return success Whether authentication succeeded
     * 
     * The signature must be over: keccak256(abi.encodePacked(nonce, address, chainId))
     */
    function authenticate(
        bytes calldata signature
    ) external nonReentrant returns (bool success) {
        UserRecord storage user = users[msg.sender];
        Challenge storage challenge = challenges[msg.sender];
        
        // Check user status
        if (!user.isActive) {
            emit AuthenticationFailed(msg.sender, "User not registered", block.timestamp);
            revert UserNotRegistered();
        }
        
        // Check challenge validity
        if (challenge.nonce == bytes32(0)) {
            emit AuthenticationFailed(msg.sender, "No challenge found", block.timestamp);
            revert ChallengeNotFound();
        }
        
        if (block.timestamp > challenge.expiryTime) {
            emit AuthenticationFailed(msg.sender, "Challenge expired", block.timestamp);
            revert ChallengeExpired();
        }
        
        if (challenge.used) {
            emit AuthenticationFailed(msg.sender, "Challenge already used", block.timestamp);
            revert ChallengeAlreadyUsed();
        }
        
        // Construct the message that should have been signed
        bytes32 messageHash = keccak256(abi.encodePacked(
            challenge.nonce,
            msg.sender,
            block.chainid
        ));
        
        // Verify signature
        if (!_verifySignature(messageHash, signature, user.publicKeyHash)) {
            emit AuthenticationFailed(msg.sender, "Invalid signature", block.timestamp);
            revert InvalidSignature();
        }
        
        // Mark challenge as used (prevents replay)
        challenge.used = true;
        usedNonces[challenge.nonce] = true;
        
        // Update statistics
        user.authCount++;
        totalAuths++;
        
        emit AuthenticationSuccess(msg.sender, block.timestamp, user.authCount);
        
        return true;
    }
    
    // =========================================================================
    // View Functions
    // =========================================================================
    
    /**
     * @notice Get user's helper data for key recovery
     * @param user Address of the user
     * @return helperData The stored helper data
     */
    function getHelperData(address user) external view returns (bytes memory) {
        if (!users[user].isActive) {
            revert UserNotRegistered();
        }
        return users[user].helperData;
    }
    
    /**
     * @notice Check if a user is registered and active
     * @param user Address to check
     * @return isRegistered Whether the user is registered and active
     */
    function isUserRegistered(address user) external view returns (bool) {
        return users[user].isActive;
    }
    
    /**
     * @notice Get user's authentication statistics
     * @param user Address of the user
     * @return registrationTime When the user registered
     * @return authCount Number of successful authentications
     */
    function getUserStats(address user) external view returns (
        uint256 registrationTime,
        uint256 authCount
    ) {
        UserRecord storage record = users[user];
        return (record.registrationTime, record.authCount);
    }
    
    /**
     * @notice Get current challenge for a user (if any)
     * @param user Address of the user
     * @return nonce Challenge nonce
     * @return expiryTime Expiry timestamp
     * @return isValid Whether the challenge is still valid
     */
    function getChallenge(address user) external view returns (
        bytes32 nonce,
        uint256 expiryTime,
        bool isValid
    ) {
        Challenge storage challenge = challenges[user];
        bool valid = !challenge.used && 
                     challenge.nonce != bytes32(0) && 
                     block.timestamp <= challenge.expiryTime;
        return (challenge.nonce, challenge.expiryTime, valid);
    }
    
    // =========================================================================
    // Internal Functions
    // =========================================================================
    
    /**
     * @notice Verify an ECDSA signature
     * @param messageHash Hash of the signed message
     * @param signature The signature bytes
     * @param expectedKeyHash Expected hash of the signer's public key
     * @return valid Whether the signature is valid
     */
    function _verifySignature(
        bytes32 messageHash,
        bytes memory signature,
        bytes32 expectedKeyHash
    ) internal pure returns (bool) {
        // Convert to Ethereum signed message hash
        bytes32 ethSignedHash = messageHash.toEthSignedMessageHash();
        
        // Recover signer address
        address signer = ethSignedHash.recover(signature);
        
        // Compute hash of recovered address (as a proxy for public key)
        // Note: In production, you might want to store the actual public key
        // and verify against that
        bytes32 signerHash = keccak256(abi.encodePacked(signer));
        
        return signerHash == expectedKeyHash;
    }
    
    // =========================================================================
    // Admin Functions
    // =========================================================================
    
    /**
     * @notice Emergency deactivation of a user (admin only)
     * @param user Address to deactivate
     * @dev Use only in case of detected fraud or key compromise
     */
    function adminDeactivate(address user) external onlyOwner {
        if (!users[user].isActive) {
            revert UserNotRegistered();
        }
        
        users[user].isActive = false;
        totalUsers--;
        
        emit UserDeactivated(user, block.timestamp);
    }
}


/**
 * @title BiometricAuthFactory
 * @notice Factory contract for deploying new BiometricAuth instances
 * @dev Useful for multi-tenant deployments or application isolation
 */
contract BiometricAuthFactory {
    event ContractDeployed(address indexed contractAddress, address indexed owner);
    
    address[] public deployedContracts;
    
    function deploy() external returns (address) {
        BiometricAuth newContract = new BiometricAuth();
        newContract.transferOwnership(msg.sender);
        
        deployedContracts.push(address(newContract));
        
        emit ContractDeployed(address(newContract), msg.sender);
        
        return address(newContract);
    }
    
    function getDeployedContracts() external view returns (address[] memory) {
        return deployedContracts;
    }
}
