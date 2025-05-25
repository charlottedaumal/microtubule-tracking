package ch.epfl.bio410.graph;


import java.awt.*;
import java.util.ArrayList;


/**
 * The Spots class represents a dynamic list of Spot objects with additional utility methods and visual attributes
 * to support tracking applications.
 */
public class Spots extends ArrayList<Spot> {

    public Color color = Color.black; // base-color for all points of the same trajectory
    public Color[] speed_color = new Color[this.size()]; // list of colors depending on their speeds, even though part of
    // the same trajectory

    /**
     * Constructor of the class. This method constructs an empty Spots list and assigns a random color.
     */
    public Spots() {
        color = Color.getHSBColor((float) Math.random(), 1f, 1f);
        color = new Color(color.getRed(), color.getGreen(), color.getBlue(), 120);
        speed_color = new Color[this.size()];
    }

    /**
     * This method initializes the color array for spot speeds to match the current size of the spot list.
     */
    public void initSpeedColor() {
        speed_color = new Color[this.size()];
    }

    /**
     * This method returns the last Spot in the list, or null if the list is empty.
     *
     * @return The most recently added Spot, or null if the list is empty.
     */
    public Spot last() {
        if (size() <= 0 ) return null;
        return get(size()-1);
    }
}
