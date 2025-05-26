package ch.epfl.bio410.graph;


/**
 * The DirectionVector class represents a 2D directional vector associated with a Spot object.
 * Here, it is used to describe the direction and speed of a moving object between frames.
 * This class provides methods to compute vector norm, dot product, and cosine similarity
 * between direction vectors, which are useful for comparing motion directions.
 */
public class DirectionVector {
    public final double dx; // x-component of the direction vector
    public final double dy; // y-component of the direction vector
    public final Spot spot; // origin of the direction vector


    /**
     * Constructor of the class.
     *
     * @param dx The x-component of the direction vector.
     * @param dy The y-component of the direction vector.
     * @param spot The Spot object from which this direction vector originates.
     */
    public DirectionVector(double dx, double dy, Spot spot) {
        this.dx = dx;
        this.dy = dy;
        this.spot = spot;
    }


    /**
     * This method computes the norm (Euclidean distance) of the direction vector.
     *
     * @return The norm of the vector.
     */
    public double norm() {
        return Math.sqrt(dx * dx + dy * dy);
    }


    /**
     * This method computes the dot product of this vector with another DirectionVector object.
     *
     * @param other The other DirectionVector to compare with.
     * @return The dot product of the two vectors.
     */
    public double dot(DirectionVector other) {
        return dx * other.dx + dy * other.dy;
    }


    /**
     * This method computes the cosine similarity between this vector and another DirectionVector.
     * The result is in the range [-1, 1], where: 1 means the vectors are perfectly aligned, 0 means the
     * vectors are orthogonal and -1 means the vectors are in opposite directions.
     * If either vector has zero magnitude, the result is defined as 0.
     *
     * @param other The other DirectionVector to compare with.
     * @return The cosine similarity between the two vectors.
     */
    public double cosineSimilarity(DirectionVector other) {
        double mag1 = norm();
        double mag2 = other.norm();
        if (mag1 == 0 || mag2 == 0) return 0; // avoid division by zero if either vector is null
        return dot(other) / (mag1 * mag2);
    }
}
