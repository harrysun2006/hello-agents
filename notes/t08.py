# 学习 neo4j
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(override=True)

NEO_URI = os.getenv("NEO4J_BOLT", "bolt://localhost:7687")
NEO_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO_DB = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USERNAME, NEO_PASSWORD))

# —— Create ——
def create_people_and_relation():
    cypher = """
    MERGE (a:Person {name: $name1})
      ON CREATE SET a.age = $age1
    MERGE (b:Person {name: $name2})
      ON CREATE SET b.age = $age2
    MERGE (a)-[r:KNOWS]->(b)
      ON CREATE SET r.since = $since, r.where = $where
    RETURN a, r, b
    """
    with driver.session(database=NEO_DB) as session:
        result = session.run(
            cypher,
            name1="Alice", age1=30,
            name2="Bob", age2=28,
            since=2024, where="苏州"
        )
        print(result.single())

# —— Read ——
def get_people_over(age):
    cypher = """
    MATCH (p:Person)
    WHERE p.age > $age
    RETURN p.name AS name, p.age AS age
    """
    with driver.session(database=NEO_DB) as session:
        return list(session.run(cypher, age=age))

# —— Update ——
def update_person_city(name, city):
    cypher = """
    MATCH (p:Person {name: $name})
    SET p.city = $city
    RETURN p
    """
    with driver.session(database=NEO_DB) as session:
        return session.run(cypher, name=name, city=city).single()

# —— Delete ——
def delete_person(name):
    cypher = """
    MATCH (p:Person {name: $name})
    DETACH DELETE p
    """
    with driver.session(database=NEO_DB) as session:
        session.run(cypher, name=name)

def main():
    create_people_and_relation()
    print(get_people_over(20))
    update_person_city("Alice", "Shanghai")
    # delete_person("Bob")
    # delete_person("Alice")
    driver.close()

if __name__ == "__main__":
    main()