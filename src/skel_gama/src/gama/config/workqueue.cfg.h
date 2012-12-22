/*
 * wqconfig.h
 *
 *  Created on: Apr 24, 2012
 *      Author: jbarbosa
 */

#ifndef WQCONFIG_H_
#define WQCONFIG_H_

const unsigned long SYSTEM_QUEUE_SIZE = 16 * 1024;
const unsigned long DEVICE_QUEUE_SIZE = 16 * 1024;
const unsigned long INBOX_QUEUE_SIZE = 1024;//NTHREAD*14*NTHREAD;
const unsigned long OUTBOX_QUEUE_SIZE = 1024;

#endif /* WQCONFIG_H_ */
