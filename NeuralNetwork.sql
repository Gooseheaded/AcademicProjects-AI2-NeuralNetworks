
DROP TABLE IF EXISTS neurons;
DROP TABLE IF EXISTS biases;
DROP TABLE IF EXISTS weights;

CREATE TABLE neurons(
	id 			INTEGER PRIMARY KEY,
	layer		INTEGER NOT NULL,
	net_input	REAL,
	output 		REAL,
	delta 		REAL
);

CREATE TABLE biases(
	layer	INTEGER PRIMARY KEY,
	value	REAL
);

CREATE TABLE weights(
	from_layer	INTEGER NOT NULL,
	from_id		INTEGER NOT NULL,
	to_layer	INTEGER NOT NULL,
	to_id		INTEGER NOT NULL,
	value 		REAL NOT NULL,
	gradient	REAL,

	PRIMARY KEY(from_layer, from_id, from_layer, to_id),
	FOREIGN KEY(from_layer) REFERENCES neurons(layer),
	FOREIGN KEY(from_id) REFERENCES neurons(id),
	FOREIGN KEY(to_layer) REFERENCES neurons(layer),
	FOREIGN KEY(to_id) REFERENCES neurons(id)
);