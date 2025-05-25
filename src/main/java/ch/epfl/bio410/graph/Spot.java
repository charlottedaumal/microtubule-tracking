package ch.epfl.bio410.graph;


/**
 * The Spot class represents a point-like object detected in an image sequence.
 */
public class Spot {
	public int x;
	public int y;
	public int t;
	public double value = 0;


	/**
	 * Constructor of the class.
	 *
	 * @param x x-coordinate.
	 * @param y y-coordinate.
	 * @param t frame number.
	 * @param value pixel intensity.
	 */
	public Spot(int x, int y, int t, double value) {
		this.x = x;
		this.y = y;
		this.t = t;
		this.value = value;
	}


	/**
	 * Method computing the Euclidean distance between this.spot and another spot.
	 *
	 * @param spot other spot.
	 * @return distance between this.spot and the other spot.
	 */
	public double distance(Spot spot){
		return Math.sqrt(Math.pow(this.x - spot.x, 2) + Math.pow(this.y - spot.y, 2));
	}
}
