package ch.epfl.bio410.graph;


import java.awt.*;
import java.util.ArrayList;

/**
 * Class implementing a "Spots" object. A "Spots" object is a list of "Spot" objects, with additional methods
 * to retrieve easily the first and the last the "Spot" of the list.
 */
public class Spots extends ArrayList<Spot> {

    public Color color = Color.black;
    public Color[] speed_color = new Color[this.size()];

    /**
     * Constructor of the class = mandatory method to build and initialize the "Spots" object
     */
    public Spots() {
        color = Color.getHSBColor((float) Math.random(), 1f, 1f);
        color = new Color(color.getRed(), color.getGreen(), color.getBlue(), 120);
        speed_color = new Color[this.size()];
    }

    public void initSpeedColor() {
        speed_color = new Color[this.size()];
    }

    public Spot last() {
        if (size() <= 0 ) return null;
        return get(size()-1);
    }

    public Spot first() {
        if (size() <= 0 ) return null;
        return get(0);
    }
}
