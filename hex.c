#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 3 // set to desired board size
#endif

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
    int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
    int open_positions[BOARD_DIM*BOARD_DIM];
    int number_of_open_positions;
    int moves[BOARD_DIM*BOARD_DIM];
    int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM+2; ++i) {
        for (int j = 0; j < BOARD_DIM+2; ++j) {
            hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
            hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

            if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
                hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
            }

            if (i == 0) {
                hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
            } else {
                hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
            }

            if (j == 0) {
                hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
            } else {
                hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
            }
        }
    }
    hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) {
    hg->connected[position*2 + player] = 1;

    if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
        return 1;
    }
    if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
        return 1;
    }

    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
            if (hg_connect(hg, player, neighbor)) {
                return 1;
            }
        }
    }
    return 0;
}

int hg_winner(struct hex_game *hg, int player, int position) {
    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->connected[neighbor*2 + player]) {
            return hg_connect(hg, player, position);
        }
    }
    return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player) {
    int random_empty_position_index = rand() % hg->number_of_open_positions;
    int empty_position = hg->open_positions[random_empty_position_index];

    if (player == 0) {
        hg->board[empty_position * 2] = 1;  // player 1
    } else {
        hg->board[empty_position * 2 + 1] = 2;  // player 2
    }

    hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;
    hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions - 1];
    hg->number_of_open_positions--;

    return empty_position;
}

int hg_full_board(struct hex_game *hg) {
    return hg->number_of_open_positions == 0;
}

void hg_print_board(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM; ++i) {
        for (int j = 0; j < i; j++) {
            printf(" ");
        }

        for (int j = 0; j < BOARD_DIM; ++j) {
            if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
                printf(" X");
            } else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 2) {
                printf(" O");
            } else {
                printf(" .");
            }
        }
        printf("\n");
    }
}
void write_csv_header(FILE *file) {
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            fprintf(file, "%d_%d,", i, j);  // use coordinates for column names
        }
    }
    fprintf(file, "Winner\n");  // winner column
}
void save_game_data(struct hex_game *hg, int winner, FILE *file) {
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            if (hg->board[(i * (BOARD_DIM + 2) + j) * 2] == 1) {
                fprintf(file, "X,");
            } else if (hg->board[(i * (BOARD_DIM + 2) + j) * 2 + 1] == 2) {
                fprintf(file, "O,");
            } else {
                fprintf(file, ".,");
            }
        }
    }
    fprintf(file, "%d\n", (winner == 1) ? 1 : 0);  
}



void save_partial_game_data(struct hex_game *hg, int winner, FILE *file, int moves_to_remove) {
    struct hex_game temp = *hg;  // copy the current game state

    // remove the last `moves_to_remove` moves, switching between players
    for (int i = 0; i < moves_to_remove && temp.number_of_open_positions < BOARD_DIM * BOARD_DIM; ++i) {
        int last_move_index = BOARD_DIM * BOARD_DIM - temp.number_of_open_positions - 1;
        int last_move = temp.moves[last_move_index];

        // determine which player's move is being removed
        if (temp.board[last_move * 2] == 1) {
            temp.board[last_move * 2] = 0;  // remove X
        } else if (temp.board[last_move * 2 + 1] == 2) {
            temp.board[last_move * 2 + 1] = 0;  // remove O
        }

        temp.number_of_open_positions++;
        temp.open_positions[temp.number_of_open_positions - 1] = last_move;
    }

    // save the resulting board
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            if (temp.board[(i * (BOARD_DIM + 2) + j) * 2] == 1) {
                fprintf(file, "X,");
            } else if (temp.board[(i * (BOARD_DIM + 2) + j) * 2 + 1] == 2) {
                fprintf(file, "O,");
            } else {
                fprintf(file, ".,");
            }
        }
    }
    fprintf(file, "%d\n", (winner == 1) ? 1 : 0); 
}

int main() {
    struct hex_game hg;

    // open three separate CSV files for the datasets
    FILE *file_complete = fopen("hex_game_data_complete.csv", "w");
    FILE *file_2_moves_before = fopen("hex_game_data_2_moves_before.csv", "w");
    FILE *file_5_moves_before = fopen("hex_game_data_5_moves_before.csv", "w");

    write_csv_header(file_complete);
    write_csv_header(file_2_moves_before);
    write_csv_header(file_5_moves_before);
//change this as boards go bigger we want boards with empty spaces cause tm then easily learns and also its more realistic boards 3x3 = 0.2-0-3 (0.2), 5x5= 0.3-0.45(0.4), 7x7 = 0.45-0.55(0.5), 9x9= 0.5-0-55(0.55) 11x11 = 0.50-65(0.5)

    #define EMPTY_CELL_THRESHOLD ((int)(BOARD_DIM * BOARD_DIM * 0.2))
    int winner = -1;
    int total_games = 10000000; 
    int target_valid_games = 2000; // the amount of games wanted for the dataset
    int valid_games = 0;
    int skipped_games = 0;
    int x_wins = 0;
    int y_wins = 0;
    int total_moves = 0;

    for (int game = 0; game < total_games; ++game) {
        hg_init(&hg);

        int player = 0;
        int moves_played = 0;

        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);
            moves_played++;
            if (hg_winner(&hg, player, position)) {
                winner = (player == 0) ? 1 : 2;
                break;
            }
            player = 1 - player;
        }

        if (hg.number_of_open_positions < EMPTY_CELL_THRESHOLD) {
            //skip if its less empty open positsions then the threshold require
            skipped_games++;
            continue;
        }
        //balance data
        if (winner == 1 && (double)x_wins / (x_wins + y_wins + 1) > 0.55) {
            skipped_games++;
            continue;
        }

        if (winner == 2 && (double)y_wins / (x_wins + y_wins + 1) > 0.55) {
            skipped_games++;
            continue;
        }

        if (winner == 1) {
            x_wins++;
        } else if (winner == 2) {
            y_wins++;
        }

        save_game_data(&hg, winner, file_complete);                           // save the completed game
        save_partial_game_data(&hg, winner, file_2_moves_before, 2);         // save two moves before
        save_partial_game_data(&hg, winner, file_5_moves_before, 5);         // save five moves before

        total_moves += moves_played;
        valid_games++;

        printf("\nValid Game %d (Player %d wins):\n", valid_games, winner);
        hg_print_board(&hg);

        if (valid_games >= target_valid_games) break;
    }

    fclose(file_complete);
    fclose(file_2_moves_before);
    fclose(file_5_moves_before);

    printf("\nTotal valid games: %d\n", valid_games);
    printf("Skipped games: %d\n", skipped_games);
    printf("Player X (1) wins: %d\n", x_wins);
    printf("Player Y (2) wins: %d\n", y_wins);
    printf("Average moves per valid game: %.2f\n", (double)total_moves / valid_games);

    return 0;
}
