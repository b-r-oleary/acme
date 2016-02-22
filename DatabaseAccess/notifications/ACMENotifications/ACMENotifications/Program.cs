using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;
using System.Data;

/***********************************/
/* this is a simple test program   */
/* testing out the C# sql interface*/
/* classes.                        */
/***********************************/

namespace ACMENotifications
{
    class Program
    {
        static void Main(string[] args)
        {
            /* open the sql client connection
            in principle, the connection string should be input,
            but in this case we want the connection to be made
            on start-up which means that we should probably embed
            the connection string in a .txt file or something */

            string connection_string;
            connection_string = "Data Source=25.70.187.150;Initial Catalog=LoggingConfigSQL;User ID=sa;Password=ThO14489";

            SqlConnection cnn = new SqlConnection(connection_string);

            try
            {
                cnn.Open();
                System.Console.WriteLine("Connection Open");
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Can not open connection");
            }

            /* main loop - wait for user to input a command, and then
            execute the command - close sql connection upon inputting
            exit or close. */

            while (true)
            {
                string input = System.Console.ReadLine();
                if (input.Equals("exit") || input.Equals("close"))
                {
                    cnn.Close();
                    System.Console.WriteLine("Connection Closed");
                    break;
                }
                else
                {
                    /* assume that the input is an SQL command -
                    issue the command to the database, and print
                    the results to the console */

                    SqlCommand cmd = new SqlCommand();
                    cmd.CommandText = input;
                    cmd.Connection = cnn;

                    try
                    {
                        SqlDataReader reader = cmd.ExecuteReader();
                        while (reader.Read())
                        {
                            for (int i = 0; i < reader.FieldCount; i++)
                            {
                                string line = reader.GetValue(i).ToString();
                                if (line.Equals(""))
                                {
                                    Console.WriteLine("NULL");
                                }
                                else
                                {
                                    Console.WriteLine(line);
                                }
                            }
                            Console.WriteLine("--------------------------");
                        }
                        reader.Close();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.ToString());
                    }
                }
            }
        }
    }

}
