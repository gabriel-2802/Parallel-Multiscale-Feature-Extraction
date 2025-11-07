#pragma once


// abstract class
class Entity {
public:
    Entity(int numtasks, int rank) : numtasks(numtasks), rank(rank) {};
    virtual ~Entity() {};
    virtual void run() {};

protected:
    const int numtasks;
    const int rank;
};