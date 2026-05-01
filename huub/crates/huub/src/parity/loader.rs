

use std::fs;

use super::game::{Game, RewardType};

///# load_game_from_dzn
///parse a .dzn file and construct a Game struct
pub fn load_game_from_dzn(path: &str) -> Game {
    let content = fs::read_to_string(path)
        .expect("failed to read dzn file");

    let nvertices = parse_scalar("nvertices", &content) as usize;
    let owners = parse_array("owners", &content)
        .into_iter().map(|x| x as u8).collect();
    let priorities = parse_array("priors", &content)
        .into_iter().map(|x| x as u32).collect();
    let sources = parse_array("sources", &content);
    let targets = parse_array("targets", &content);
    let weights = parse_array("weights", &content);

    //construct Game using existing constructor
    Game::new(
        nvertices,
        owners,
        priorities,
        sources,
        targets,
        weights,
        0,                  //init (not used)
        RewardType::Min,    //default
    )
}

///# parse_scalar
///extract single integer value from dzn
fn parse_scalar(name: &str, content: &str) -> i32 {
    let line = find_line(name, content);
    line.split('=')
        .nth(1).unwrap()
        .replace(';', "")
        .trim()
        .parse()
        .expect("failed to parse scalar")
}

///# parse_array
///extract array [a,b,c] from dzn
fn parse_array(name: &str, content: &str) -> Vec<i32> {
    let line = find_line(name, content);

    let inside = line.split('[')
        .nth(1).unwrap()
        .split(']')
        .next().unwrap();

    inside.split(',')
        .map(|x| x.trim().parse::<i32>().unwrap())
        .collect()
}

///# find_line
///find the line starting with given key
fn find_line<'a>(name: &str, content: &'a str) -> &'a str {
    content.lines()
        .find(|l| l.trim().starts_with(name))
        .expect("field not found in dzn")
}