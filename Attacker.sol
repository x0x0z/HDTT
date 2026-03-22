// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IDemo {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
}

contract Attacker {
    IDemo public target;

    constructor(address _target) {
        target = IDemo(_target);
    }

    // Atomic Attack: Deposit and Withdraw in the SAME transaction
    // This generates "Internal Transactions" which are a key feature for ML
    function attack() external payable {
        target.deposit{value: msg.value}();
        target.withdraw(msg.value);
    }
    
    // Allow receiving ETH back
    receive() external payable {}
}