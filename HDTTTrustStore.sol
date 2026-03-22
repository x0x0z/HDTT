// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract HDTTTrustStore {
    address public oracle; 
    mapping(address => uint256) public trustScore; 
    mapping(address => uint256) public lastUpdate; 

    event TrustUpdated(address indexed user, uint256 newScore, uint256 timestamp);

    constructor() {
        oracle = msg.sender;
    }

    modifier onlyOracle() {
        require(msg.sender == oracle, "Only oracle");
        _;
    }

    function setOracle(address _newOracle) external onlyOracle {
        oracle = _newOracle;
    }

    function updateTrustScore(address user, uint256 score) external onlyOracle {
        require(score <= 100, "Score > 100");
        trustScore[user] = score;
        lastUpdate[user] = block.timestamp;
        emit TrustUpdated(user, score, block.timestamp);
    }

    function getTrustScore(address user) public view returns (uint256) {
        // Default score for unknown users is 50 (Neutral)
        if (lastUpdate[user] == 0) {
            return 50; 
        }
        return trustScore[user];
    }
}