extern crate rand;
extern crate rayon;
extern crate bytecount;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<u8>,
    width: usize,
    height: usize,
    length: usize,
    min_surv: u8,
    max_surv: u8,
    min_birth: u8,
    max_birth: u8,
    max_age: u8
}

impl CellA {
    pub fn new ( width: usize, height: usize, length: usize, min_surv: u8, max_surv: u8, min_birth: u8, max_birth: u8 ) -> Self {
        let cells = vec![0; width * height * length];
        let max_age = 5;

        Self {
            cells,
            width,
            height,
            length,
            min_surv,
            max_surv,
            min_birth,
            max_birth,
            max_age,
        }
    }

    pub fn set_xyz ( &mut self, x: usize, y: usize, z: usize, new_state: u8 ) {
        let idx = (z * self.width * self.height) + (y * self.width) + x;
        self.cells[idx] = new_state;
    }

    pub fn next_gen ( &mut self ) {
        let new_cells = (0 .. self.width * self.height * self.length)
            .into_par_iter()
            .map(|idx| {
                if (idx > self.width * self.height + self.width) && (idx < (self.width * self.height * self.length) - (self.width * self.height) - self.width - 1) {
                    let cur_state = self.cells[idx];
                    if cur_state >= self.max_age {
                        return 0
                    }

                    let neighbors = [
                        self.cells[idx + (self.width * self.height) + self.width + 1],
                        self.cells[idx + (self.width * self.height) + self.width    ],
                        self.cells[idx + (self.width * self.height) + self.width - 1],
                        self.cells[idx + (self.width * self.height)              + 1],
                        self.cells[idx + (self.width * self.height)                 ],
                        self.cells[idx + (self.width * self.height)              - 1],
                        self.cells[idx + (self.width * self.height) - self.width + 1],
                        self.cells[idx + (self.width * self.height) - self.width    ],
                        self.cells[idx + (self.width * self.height) - self.width - 1],
                        self.cells[idx                              + self.width + 1],
                        self.cells[idx                              + self.width    ],
                        self.cells[idx                              + self.width - 1],
                        self.cells[idx                                           + 1],
                        self.cells[idx                                           - 1],
                        self.cells[idx                              - self.width + 1],
                        self.cells[idx                              - self.width    ],
                        self.cells[idx                              - self.width - 1],
                        self.cells[idx - (self.width * self.height) + self.width + 1],
                        self.cells[idx - (self.width * self.height) + self.width    ],
                        self.cells[idx - (self.width * self.height) + self.width - 1],
                        self.cells[idx - (self.width * self.height)              + 1],
                        self.cells[idx - (self.width * self.height)                 ],
                        self.cells[idx - (self.width * self.height)              - 1],
                        self.cells[idx - (self.width * self.height) - self.width + 1],
                        self.cells[idx - (self.width * self.height) - self.width    ],
                        self.cells[idx - (self.width * self.height) - self.width - 1]
                    ];

                    // let count = bytecount::count(&neighbors, 1) as u8;
                    let count: u8 = neighbors.iter().sum();

                    if cur_state > 0 {
                        if count >= self.min_surv && count <= self.max_surv {
                            cur_state + 1
                        } else {
                            0
                        }
                    } else if count >= self.min_birth && count <= self.max_birth {
                        self.max_age
                    } else {
                        0
                    }
                } else {
                    0
                }
            })
            .collect();

        self.cells = new_cells;
    }
}
