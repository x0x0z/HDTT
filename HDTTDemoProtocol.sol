// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./HDTTTrustStore.sol";

contract HDTTDemoProtocol {
    HDTTTrustStore public trustStore;
    
    // Tracks internal user balances (Fixes the Native ETH vs Internal Balance bug)
    mapping(address => uint256) public balances;
    mapping(address => uint256) public lastWithdrawTime;

    event Deposit(address indexed user, uint256 amount);
    event Withdraw(address indexed user, uint256 amount);
    event SecurityBlock(address indexed user, string reason, uint256 score);

    constructor(address _trustStoreAddress) {
        trustStore = HDTTTrustStore(_trustStoreAddress);
    }

    // --- Dynamic Limit Logic ---

    function getMaxWithdraw(address user) public view returns (uint256) {
        uint256 score = trustStore.getTrustScore(user);
        if (score >= 80) return 100 ether;  // High Trust
        if (score >= 60) return 50 ether;   // Medium Trust
        if (score >= 40) return 10 ether;   // Low Trust
        if (score >= 20) return 1 ether;    // Suspicious
        return 0 ether;                     // Blocked
    }

    function getCooldown(address user) public view returns (uint256) {
        uint256 score = trustStore.getTrustScore(user);
        if (score >= 80) return 0;           // Instant
        if (score >= 60) return 1 minutes;
        if (score >= 40) return 1 hours;
        return 24 hours;
    }

    // --- Core Actions ---

    function deposit() external payable {
        require(msg.value > 0, "Zero deposit");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        address user = msg.sender;
        uint256 score = trustStore.getTrustScore(user);

        // 1. Check Internal Balance
        require(balances[user] >= amount, "Insufficient internal balance");

        // 2. Check Trust Limits
        uint256 maxLimit = getMaxWithdraw(user);
        if (amount > maxLimit) {
            emit SecurityBlock(user, "Exceeds Trust Limit", score);
            revert("HDTT: Exceeds trust-based limit");
        }

        // 3. Check Cooldown
        uint256 cooldown = getCooldown(user);
        if (block.timestamp < lastWithdrawTime[user] + cooldown) {
            emit SecurityBlock(user, "Cooldown Active", score);
            revert("HDTT: Cooldown not passed");
        }

        // 4. Execute
        balances[user] -= amount;
        lastWithdrawTime[user] = block.timestamp;
        
        (bool sent, ) = payable(user).call{value: amount}("");
        require(sent, "ETH Transfer failed");
        
        emit Withdraw(user, amount);
    }
    
    // Helper to check contract solvency
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}