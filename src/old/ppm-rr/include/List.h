#ifndef LIST_LNK_H
#define LIST_LNK_H

//#include "core.h"

/**
 *     for(domain->childs.gotoBeginning();domain->childs.canStep();domain->childs.step()){
 X
 }
 */

using namespace std;

template<class DT>
class LList;

template<class DT>
class ListNode {
private:

  DT dataItem;
  ListNode *next;

  friend class LList<DT> ;

  __H_D__ ListNode( DT &nodeData, ListNode *nextPtr) {

    dataItem = nodeData;
    next = nextPtr;
  }
};

template<class DT>
class LList {

private:

  ListNode<DT> *head, // Pointer to the beginning of the list
      *cursor; // Cursor pointer


  __H_D__
  bool isEmpty() ;__H_D__
  void gotoEnd();

public:
  pthread_mutex_t m_lock;
  int length;

  __H_D__ LList();
  __H_D__ ~LList();

  __H_D__
  void insert(DT newData);

  __H_D__
  void removeCursor_();

  __H_D__
  void inline gotoBeginning();

  __H_D__
  inline ListNode<DT>* getIterator();

  __H_D__
  bool inline gotoNext();

  __H_D__
  void inline lock();

  __H_D__
  void inline unlock();

  __H_D__
  void inline stopStep_();

  __H_D__
  bool canStep(ListNode<DT>*& local_cursor);

  __H_D__
  bool step(ListNode<DT>*& local_cursor);

  __H_D__
  inline DT getCursorValue(ListNode<DT>*& local_cursor);

  __H_D__
  void inline showStructure();

  __H_D__
  DT inline getCursor_();

  __H_D__
  void inline setCursor_(ListNode<DT> *c);

  __H_D__
  bool inline canStep_();

  __H_D__
  bool inline step_();

  __H_D__
  inline ListNode<DT>* getCursorBox_();

};

template<class DT>
__H_D__ LList<DT>::LList() {

  head = NULL;
  cursor = NULL;

  length = 0;
#ifndef __CUDACC__
  pthread_mutex_init(&m_lock, NULL);
#endif
}

template<class DT>
__H_D__ LList<DT>::~LList() {

#ifndef __CUDACC__
  //  ListNode<DT> *tmp;
  //
  //  cursor = NULL;
  //  while (head != NULL) {
  //    tmp = head;
  //    head = head->next;
  //    delete tmp;
  //  }
#endif

}

template<class DT>
__H_D__
void LList<DT>::lock() {
#ifndef __CUDACC__
  pthread_mutex_lock(&m_lock);
#endif
}

template<class DT>
__H_D__
void LList<DT>::unlock() {
#ifndef __CUDACC__
  pthread_mutex_unlock(&m_lock);
#endif
}

// Insert after cursor
template<class DT>
__H_D__ void LList<DT>::insert(DT newData) {

  lock();

  gotoEnd();

  // make new node
  ListNode<DT> *newNode = new ListNode<DT> (newData, NULL);

  // if the list is empty then just add the node
  if (head == NULL)
  head = newNode;
  // if the cursor is at the end of the list then add it to end of list
  else if (cursor->next == NULL)
  cursor->next = newNode;
  else // if the cursor is in the middle of the end then add the item, but change
  { // pointers to include the new node
    ListNode<DT> *tmp = cursor->next;
    newNode->next = tmp;
    cursor->next = newNode;
  }
  // move the cursor to the new node
  cursor = newNode;
  length++;
  //gotoBeginning();
  cursor = head;

  unlock();

}

template<class DT>
void
__H_D__
LList<DT>::removeCursor_() {

  if (isEmpty()) {
    return;
  }

  ListNode<DT>* tmp = NULL; // make new temp pointer

  if (cursor == head) { // if cursor is at the start of the list
    tmp = head;
    head = head->next;
    cursor = head;
  } else { // if cursor is some where in the list
    tmp = cursor;
    cursor = head;
    // find previous node
    while (cursor->next != tmp)
    cursor = cursor->next;
    // fix the list to skip the node in question
    cursor->next = tmp->next;

  }

  // destroy the node
  delete tmp;
  // if the node deleted was at end of list, then move to front otherwise move to next node
  if (cursor == NULL)
  cursor = head;
  else
  cursor = cursor->next;

  length--;
}

template<class DT>
__H_D__ inline bool LList<DT>::isEmpty()  {

  return (head == NULL);

}

template<class DT>
__H_D__ inline void LList<DT>::gotoBeginning() {

  if (isEmpty()) {

    return;
  }

  cursor = head;

} // Go to beginning


template<class DT>
__H_D__ inline ListNode<DT>* LList<DT>::getIterator() {

  lock();

  if (isEmpty()){
    return NULL;

  }

  ListNode<DT>* r = head;

  unlock();

  return r;
} // Go to beginning

template<class DT>
__H_D__ inline void LList<DT>::gotoEnd() {

  if (isEmpty()) {

    return;
  }

  if (cursor == NULL)
  cursor = head;
  while (cursor->next != NULL)

  cursor = cursor->next;

} // Go to end

template<class DT>
__H_D__ inline bool LList<DT>::gotoNext() {

  if (isEmpty()) {

    return false;
  }

  if (cursor->next == NULL) {

    return false;
  }

  cursor = cursor->next;

  return true;
}

template<class DT>
__H_D__ inline void LList<DT>::stopStep_() {

  cursor = NULL;

}

template<class DT>
__H_D__ inline bool LList<DT>::step_() {

  if (cursor == NULL) {

    return false;
  }

  cursor = cursor->next;

  return true;
}

template<class DT>
__H_D__ inline bool LList<DT>::canStep_() {

  if (cursor == NULL) {

    return false;
  }

  return true;
}

template<class DT>
__H_D__ inline bool LList<DT>::canStep(ListNode<DT>*& local_cursor) {
  if (local_cursor == NULL)
  return false;
  return true;
}

template<class DT>
__H_D__ inline bool LList<DT>::step(ListNode<DT>*& local_cursor) {

  if (local_cursor == NULL)
  return false;

  local_cursor = local_cursor->next;
  return true;
}

// Output the list structure -- used in testing/debugging
// Outputs the items in a list. If the list is empty, outputs
// "Empty list". This operation is intended for testing and
// debugging purposes only.
template<class DT>
__H_D__ inline void LList<DT>::showStructure() {

  ListNode<DT> *p; // Iterates through the list

  if (head == 0)
  cout << "Empty list" << endl;
  else {
    for (p = head; p != NULL; p = p->next)
    if (p == cursor)
    cout << "[" << p->dataItem << "] ";
    else
    cout << p->dataItem << " ";
    cout << endl;
  }

}

template<class DT>
__H_D__ inline DT LList<DT>::getCursor_() {

  DT* a;// = DT();
  //if (isEmpty()) {

    //return a;
  //}

  if (cursor!= NULL) a = &(cursor->dataItem);
  else {
    printf("ERROR: cursor null\n");
    //showStructure();
  }

  return *a;
}

template<class DT>
__H_D__ inline DT LList<DT>::getCursorValue(ListNode<DT>*& local_cursor) {

  DT* a;

  if (local_cursor!= NULL)
  a = &(local_cursor->dataItem);
  else {
    printf("ERROR: cursor null\n");
  }

  return *a;
}

template<class DT>
__H_D__ inline ListNode<DT>* LList<DT>::getCursorBox_() {

  ListNode<DT>* a = cursor;

  return a;
}

template<class DT>
__H_D__ inline void LList<DT>::setCursor_(ListNode<DT> *c) {

  cursor = c;

}

#endif
