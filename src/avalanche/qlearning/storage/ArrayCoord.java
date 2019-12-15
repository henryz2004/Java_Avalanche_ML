package avalanche.qlearning.storage;

public class ArrayCoord {

    public int row;
    public int col;

    public ArrayCoord(int[] coordinate) {
        row = coordinate[0];
        col = coordinate[1];
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof ArrayCoord)) return false;

        return row == ((ArrayCoord) other).row && col == ((ArrayCoord) other).col;
    }

    @Override
    public int hashCode() {
        // https://stackoverflow.com/questions/22826326/good-hashcode-function-for-2d-coordinates
        //return ((row + col)*(row + col + 1)/2) + col;
        int hashCode = 2;
        hashCode = (hashCode * 33) ^ row;
        hashCode = (hashCode * 33) ^ col;
        return hashCode;
    }
}
