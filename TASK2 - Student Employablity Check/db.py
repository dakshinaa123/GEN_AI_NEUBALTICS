# db.py
import sqlite3
from sqlite3 import Error

DB_NAME = "submissions.db"

def create_connection(db_file=DB_NAME):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table():
    """Create the submissions table if it doesn't exist."""
    conn = create_connection()
    if conn is not None:
        create_table_sql = """ 
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                degree TEXT,
                course TEXT,
                general_appearance INTEGER,
                manner_speaking INTEGER,
                mental_alertness INTEGER,
                self_confidence INTEGER,
                present_ideas INTEGER,
                comm_skills INTEGER,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()
    else:
        print("Error! Cannot create the database connection.")

def insert_submission(name, degree, course, general_appearance, manner_speaking,
                      mental_alertness, self_confidence, present_ideas, comm_skills, result):
    """Insert a new submission record into the submissions table."""
    conn = create_connection()
    sql = ''' 
    INSERT INTO submissions(name, degree, course, general_appearance, manner_speaking, 
                            mental_alertness, self_confidence, present_ideas, comm_skills, result)
    VALUES(?,?,?,?,?,?,?,?,?,?) 
    '''
    cur = conn.cursor()
    cur.execute(sql, (name, degree, course, general_appearance, manner_speaking, 
                      mental_alertness, self_confidence, present_ideas, comm_skills, result))
    conn.commit()
    conn.close()
