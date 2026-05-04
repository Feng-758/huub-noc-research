use std::{fmt::Debug, fs, path::Path, str::FromStr};

use super::game::{Game, RewardType};

///# load_game_from_dzn
///parse a .dzn file and construct a Game struct
pub fn load_game_from_dzn(path: &Path) -> Game {
	let content = fs::read_to_string(path).expect("failed to read dzn file");

	let _ = parse_scalar("nvertices", &content) as usize;
	let owners = parse_array("owners", &content);
	let priorities = parse_array("priors", &content);
	let sources: Vec<usize> = parse_array::<i64>("sources", &content)
		.into_iter()
		.map(|x| (x - 1) as usize)
		.collect();
	let targets: Vec<usize> = parse_array::<i64>("targets", &content)
		.into_iter()
		.map(|x| (x - 1) as usize)
		.collect();
	let weights = if content.contains("weights") {
		parse_array("weights", &content)
	} else {
		vec![0; sources.len()]
	};

	//construct Game using existing constructor
	Game::new(
		owners,
		priorities,
		sources,
		targets,
		weights,
		0,               //init (not used)
		RewardType::Min, //default
	)
}

///# parse_scalar
///extract single integer value from dzn
fn parse_scalar(name: &str, content: &str) -> i32 {
	let line = find_line(name, content);
	line.split('=')
		.nth(1)
		.unwrap()
		.replace(';', "")
		.trim()
		.parse()
		.expect("failed to parse scalar")
}

fn parse_array<T: FromStr>(name: &str, content: &str) -> Vec<T>
where
	T::Err: Debug,
{
	let mut collecting = false;
	let mut buffer = String::new();

	for line in content.lines() {
		let line = line.trim();
		if !collecting {
			if line.starts_with(name) {
				collecting = true;

				if let Some(idx) = line.find('[') {
					buffer.push_str(&line[idx + 1..]);
					buffer.push(' ');
				}
			}
		} else {
			buffer.push_str(line);
			buffer.push(' ');
			if line.contains(']') {
				break;
			}
		}
	}
	let inside = buffer.split(']').next().expect("missing closing ]");
	inside
		.split(',')
		.filter(|x| !x.trim().is_empty())
		.map(|x| x.trim().parse::<T>().unwrap())
		.collect()
}

///# find_line
///find the line starting with given key
fn find_line<'a>(name: &str, content: &'a str) -> &'a str {
	content
		.lines()
		.find(|l| l.trim().starts_with(name))
		.expect("field not found in dzn")
}
