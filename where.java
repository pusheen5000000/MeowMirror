import java.util.*;
import java.io.*;

public class Main {
    public static void main(String[] args){
        Scanner s = new Scanner(System.in);
        int tee = 0;
        int ess = 0;
        int number = Integer.parseInt(s.nextLine());
        for (int i=0; i<number; i++){
            String sentence = s.nextLine();
            for (int r=0; r< sentence.length(); r++){
                if (Character.toLowerCase(sentence.charAt(r)) == 't')
                    tee++;
                else if (Character.toLowerCase(sentence.charAt(r))=='s')
                    ess++;
            }
        }
    
        if (tee > ess)
            Syst    em.out.println("English");
        else
            System.out.println("French");
    }
}
