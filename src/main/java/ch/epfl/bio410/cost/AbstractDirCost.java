package ch.epfl.bio410.cost;


import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;


/**
 * The AbstractDirCost interface defines a contract for evaluating directional costs
 * between two Spot objects in the context of a PartitionedGraph, here used in tracking.
 * It provides three methods. One validation method that checks whether the connection between two spots is valid.
 * Two evaluation methods that compute a direction-based cost between two spots, one incorporating speed into the cost
 * evaluation and not the other.
 * Implementing class (DirectionCost) should define specific behaviors for these methods.
 */
public interface AbstractDirCost {

    public abstract double evaluate(Spot a, Spot b, PartitionedGraph frames, int dimension);
    public abstract double evaluate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension);
    public abstract boolean validate(Spot a, Spot b, PartitionedGraph frames, int dimension);
    public abstract boolean validate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension);

}
