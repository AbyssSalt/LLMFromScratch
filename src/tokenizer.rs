use std::{collections::{HashMap, HashSet}, fs::File, io::Read};

fn generate_vocabulary(filename: &String) -> (HashMap<char, u8>, HashMap<u8, char>){
    let mut id_char_map: HashMap<u8, char> = HashMap::new();
    let mut char_id_map: HashMap<char, u8> = HashMap::new();

    let mut file:File = std::fs::File::open(filename).expect("File not found!");
    let mut data:String = String::new();
    file.read_to_string(&mut data).expect("Non UTF-8 character found!");
    
    let mut distinct:u8 = 1;

    for character in data.chars() {
        if !char_id_map.contains_key(&character) {
            char_id_map.insert(character, distinct);
            id_char_map.insert(distinct, character);
            distinct += 1;
        }
    }

    return (char_id_map, id_char_map);
}

fn encode(text: &String, id_char_map: &HashMap<char, u8>) -> Vec<u8> {
    let mut ids: Vec<u8> = Vec::new();
    
    for char in text.chars() {
        ids.push(*id_char_map.get(&char).expect("Character not in vocabulary!"));
    }

    return ids;
}

fn decode(ids: &Vec<u8>, char_id_map: &HashMap<u8, char>) -> String {
    let mut text: String = String::new();
    
    for id in ids {
        text.push(*char_id_map.get(&id).expect("Character not in vocabulary!"));
    }

    return text;
}

pub fn main() {
    let maps: (HashMap<char, u8>, HashMap<u8, char>) = generate_vocabulary(&String::from("D:\\Visual Studio\\Rust\\LLM\\Data Parser\\CleanedData\\Hestle.txt"));
    let char_id_map: HashMap<char, u8> = maps.0;
    let id_char_map: HashMap<u8, char> = maps.1;
    let text: String = String::from("Hello");
}