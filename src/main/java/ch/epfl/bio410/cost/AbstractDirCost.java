package ch.epfl.bio410.cost;

import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;

/**
 * This is a so-called interface. To make it simple, an interface is a special class where you declare methods
 * without implementing them.
 * The goal of the interface is to be as generic as possible to be implemented in other classes.
 *
 * Here, AbstractCost declares 2 methods that are common to any cost algorithms. These methods are implemented in the
 * classes that implement this interface (DistanceAndIntensityCost and SimpleDistanceCost). Their implementation in the
 * respective classes can be different.
 */
public interface AbstractDirCost {

    public abstract boolean validate(Spot a, Spot b, PartitionedGraph frames, int dimension);
    public abstract double evaluate(Spot a, Spot b, PartitionedGraph frames, int dimension);
    public abstract double evaluate_withSpeed(Spot a, Spot b, PartitionedGraph frames, int dimension);


}
