// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract HDTTTrustStore {

    mapping(address => uint256) public trustScore;           
    mapping(address => uint256) public lastUpdateTimestamp;  
    mapping(address => uint256) public volatilityCounter;    
    
    address public owner;
    address public orchestrator;
    
    
    uint256 public constant LAMBDA = 9000;        
    uint256 public constant BETA = 1000;          
    uint256 public constant SCORE_TTL = 300;      
    uint256 public constant DEFAULT_SCORE = 50;
    uint256 public constant DECAY_BLOCK = 10000;
    
   
    event ScoreUpdated(address indexed user, uint256 newScore, uint256 decayedScore, uint256 timestamp);
    event ParametersChanged(uint256 lambda, uint256 beta, uint256 timestamp);
    
    modifier onlyOrchestrator() {
        require(msg.sender == orchestrator, "Not orchestrator");
        _;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor(address _orchestrator) {
        owner = msg.sender;
        orchestrator = _orchestrator;
    }
    
   
    // S_t(a) = λ * S_{t-1}(a) + (1-λ) * s_t
    function updateScore(address user, uint256 newScore) 
        external 
        onlyOrchestrator 
    {
        require(newScore <= 100, "Invalid score");
        
        // 衰减计算
        uint256 oldScore = trustScore[user];
        uint256 decayedScore = oldScore;
        
        if (oldScore > 0) {
            uint256 blocksSince = block.number - lastUpdateTimestamp[user];
            if (blocksSince > DECAY_BLOCK) {
                decayedScore = DEFAULT_SCORE;
            } else if (blocksSince > 0) {
                uint256 decayPercent = (blocksSince * 100) / DECAY_BLOCK;
                if (decayPercent < 100) {
                    decayedScore = (oldScore * (100 - decayPercent)) / 100;
                }
            }
        } else {
            decayedScore = DEFAULT_SCORE;
        }
        
        
        uint256 updatedScore = (decayedScore * LAMBDA + newScore * (10000 - LAMBDA)) / 10000;
        
        
        if (oldScore > 0) {
            uint256 diff = oldScore > updatedScore ? oldScore - updatedScore : updatedScore - oldScore;
            if (diff > 5) {
                volatilityCounter[user]++;
            } else {
                volatilityCounter[user] = 0;
            }
        }
        
        
        trustScore[user] = updatedScore;
        lastUpdateTimestamp[user] = block.number;
        
        emit ScoreUpdated(user, updatedScore, decayedScore, block.timestamp);
    }
    
    
    function getTrustScore(address user) 
        external 
        view 
        returns (uint256 score, bool isFresh) 
    {
        if (lastUpdateTimestamp[user] == 0) {
            return (DEFAULT_SCORE, false);
        }
        
        uint256 timestamp = lastUpdateTimestamp[user];
        if (block.timestamp > timestamp + SCORE_TTL) {
            return (DEFAULT_SCORE / 2, false);
        }
        
        return (trustScore[user], true);
    }
    
   
    // τ_t(a) = τ_0 + β * ν(a,t)
    function getAdaptiveThreshold(address user, uint256 baseThreshold) 
        external 
        view 
        returns (uint256) 
    {
        uint256 volatility = volatilityCounter[user];
        uint256 adaptiveThreshold = baseThreshold + (volatility * BETA) / 10000;
        
        if (adaptiveThreshold > 100) {
            adaptiveThreshold = 100;
        }
        
        return adaptiveThreshold;
    }
    
    
    function batchUpdateScores(address[] calldata users, uint256[] calldata scores) 
        external 
        onlyOrchestrator 
    {
        require(users.length == scores.length, "Length mismatch");
        require(users.length <= 100, "Too many");
        
        for (uint i = 0; i < users.length; i++) {
            _internalUpdate(users[i], scores[i]);
        }
    }
    
    function _internalUpdate(address user, uint256 newScore) internal {
        require(newScore <= 100, "Invalid score");
        
        uint256 oldScore = trustScore[user];
        uint256 decayedScore = oldScore > 0 ? oldScore : DEFAULT_SCORE;
        
        uint256 updatedScore = (decayedScore * LAMBDA + newScore * (10000 - LAMBDA)) / 10000;
        
        trustScore[user] = updatedScore;
        lastUpdateTimestamp[user] = block.number;
    }
    
    
    function getFullStatus(address user) 
        external 
        view 
        returns (uint256 score, uint256 volatility, uint256 lastBlock) 
    {
        return (trustScore[user], volatilityCounter[user], lastUpdateTimestamp[user]);
    }
    
    
    function setOrchestrator(address _orchestrator) external onlyOwner {
        orchestrator = _orchestrator;
    }
}
